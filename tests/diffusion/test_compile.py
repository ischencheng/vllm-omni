# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch.nn as nn

import vllm_omni.diffusion.compile as compile_module
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _WrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False
        self.forward_compiled = False

    def compile(self, *args, **kwargs):
        self.compile_called = True
        return self

    def forward(self, x):
        return x


class _ModelWithWrappedRepeatedBlocks(nn.Module):
    _repeated_blocks = ["OriginalBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_WrappedBlock(), _WrappedBlock()])
        self.other_blocks = nn.ModuleList([_WrappedBlock()])


def test_regionally_compile_matches_wrapped_blocks_by_declared_container_attr(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def _compiled(*fn_args, **fn_kwargs):
            return f"compiled:{fn(*fn_args, **fn_kwargs)}"

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model, dynamic=True)

    assert len(compile_calls) == 2
    assert all(not block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called
    assert model.transformer_blocks[0].forward("ok") == "compiled:ok"


@pytest.mark.parametrize("with_hooks", [False, True])
def test_regionally_compile_does_not_partially_mutate_on_setup_failure(monkeypatch, with_hooks):
    model = _ModelWithWrappedRepeatedBlocks()
    hooks = []
    if with_hooks:
        for block in model.transformer_blocks:
            hook = ModelHook()
            HookRegistry.get_or_create(block).register_hook("test", hook)
            hooks.append(hook)
    original_forwards = [block.forward for block in model.transformer_blocks]
    original_computes = [getattr(block, "_omni_original_forward", None) for block in model.transformer_blocks]
    compiled_blocks = {id(model.other_blocks[0])}
    compile_calls = 0

    def _compile(fn, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 2:
            raise RuntimeError("compile setup failed")
        return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs)

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    with pytest.raises(RuntimeError, match="compile setup failed"):
        regionally_compile(model, compiled_blocks=compiled_blocks, dynamic=True)

    assert [block.forward for block in model.transformer_blocks] == original_forwards
    assert [getattr(block, "_omni_original_forward", None) for block in model.transformer_blocks] == original_computes
    assert compiled_blocks == {id(model.other_blocks[0])}
    for hook, original_compute in zip(hooks, original_computes):
        assert hook.fn_ref.original_forward is original_compute


def test_regionally_compile_keeps_hook_dispatch_outside_compiled_graph(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    block = model.transformer_blocks[0]
    registry = HookRegistry.get_or_create(block)
    hook = ModelHook()
    registry.register_hook("test", hook)

    wrapped_forward = block.forward
    original_forward = block._omni_original_forward
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append(fn)

        def _compiled(*fn_args, **fn_kwargs):
            return f"compiled:{fn(*fn_args, **fn_kwargs)}"

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model)

    assert compile_calls[0] is original_forward
    assert compile_calls[0] is not wrapped_forward
    assert block.forward is wrapped_forward
    assert block._omni_original_forward is not original_forward
    assert hook.fn_ref.original_forward is block._omni_original_forward
    assert block("ok") == "compiled:ok"


def test_compiled_block_preserves_forward_signature_for_inspection(monkeypatch):
    """cache-dit matches blocks via inspect.signature(block.forward).

    Whatever regionally_compile installs as the block forward must stay
    signature-transparent: parameter names and the return annotation drive
    cache-dit's ForwardPattern match (build 2954, Multi-GPU Layered job
    failed when a bare *args/**kwargs wrapper hid them; torch.compile's own
    wrapper preserves the signature).
    """
    import inspect

    import torch

    class _SignatureBlock(nn.Module):
        def forward(self, hidden_states, encoder_hidden_states=None) -> "torch.Tensor":
            return hidden_states

    class _Model(nn.Module):
        _repeated_blocks = ["_SignatureBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_SignatureBlock()])

    model = _Model()
    monkeypatch.setattr(compile_module.torch, "compile", lambda fn, *a, **k: fn)
    regionally_compile(model)

    sig = inspect.signature(model.blocks[0].forward)
    assert set(sig.parameters.keys()) == {"hidden_states", "encoder_hidden_states"}
    assert "torch.Tensor" in str(sig.return_annotation)
    assert model.blocks[0].forward("x") == "x"


@pytest.mark.parametrize("with_hooks", [False, True])
def test_regionally_compile_deduplicates_blocks_shared_across_roots(monkeypatch, with_hooks):
    first_model = _ModelWithWrappedRepeatedBlocks()
    second_model = _ModelWithWrappedRepeatedBlocks()
    block = first_model.transformer_blocks[0]
    second_model.transformer_blocks[0] = block
    if with_hooks:
        hook = ModelHook()
        HookRegistry.get_or_create(block).register_hook("test", hook)
        wrapped_forward = block.forward
        original_forward = block._omni_original_forward
    else:
        original_forward = block.forward
    compile_calls = []
    compiled_blocks: set[int] = set()

    def compile_forward(fn, *args, **kwargs):
        compile_calls.append(fn)
        return lambda x: f"compiled:{fn(x)}"

    monkeypatch.setattr(compile_module.torch, "compile", compile_forward)

    regionally_compile(first_model, compiled_blocks=compiled_blocks)
    regionally_compile(second_model, compiled_blocks=compiled_blocks)

    assert len(compile_calls) == 3
    assert compile_calls.count(original_forward) == 1
    assert compiled_blocks == {
        id(block),
        id(first_model.transformer_blocks[1]),
        id(second_model.transformer_blocks[1]),
    }
    assert block("ok") == "compiled:ok"
    if with_hooks:
        assert block.forward is wrapped_forward
        assert hook.fn_ref.original_forward is block._omni_original_forward


def test_regionally_compile_real_backend_with_shared_hooked_block():
    import torch

    class TensorBlock(nn.Module):
        def forward(self, x):
            return x.sin() + 1

    model = _ModelWithWrappedRepeatedBlocks()
    block = TensorBlock()
    model.transformer_blocks[0] = block
    hook = ModelHook()
    HookRegistry.get_or_create(block).register_hook("test", hook)
    wrapped_forward = block.forward
    compiled_blocks: set[int] = set()
    values = torch.arange(4, dtype=torch.float32)
    expected = block(values)
    graphs = []

    def eager_backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    regionally_compile(model, compiled_blocks=compiled_blocks, backend=eager_backend, fullgraph=True)
    compiled_forward = block._omni_original_forward
    regionally_compile(model, compiled_blocks=compiled_blocks, backend=eager_backend, fullgraph=True)

    assert block.forward is wrapped_forward
    assert block._omni_original_forward is compiled_forward
    assert hook.fn_ref.original_forward is compiled_forward
    torch.testing.assert_close(block(values), expected, rtol=0, atol=0)
    assert len(graphs) == 1
