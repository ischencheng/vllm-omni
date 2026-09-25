# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_dynamic_caches_preserves_values_and_ownership(batch_size, dtype, monkeypatch):
    config = Qwen3TTSTokenizerV2DecoderConfig(num_hidden_layers=2, sliding_window=8)
    caches = [DynamicCache(config=config) for _ in range(batch_size)]
    for row, cache in enumerate(caches):
        keys = torch.full((1, 2, 3, 4), row + 1, dtype=dtype)
        cache.update(keys, keys + 10, 0)
    original_keys = [cache.layers[0].keys.clone() for cache in caches]
    original_values = [cache.layers[0].values.clone() for cache in caches]
    kv_ids = {id(tensor) for cache in caches for tensor in (cache.layers[0].keys, cache.layers[0].values)}
    copied_tensor_ids = []
    tensor_deepcopy = torch.Tensor.__deepcopy__

    def record_tensor_deepcopy(tensor, memo):
        copied_tensor_ids.append(id(tensor))
        return tensor_deepcopy(tensor, memo)

    monkeypatch.setattr(torch.Tensor, "__deepcopy__", record_tensor_deepcopy)
    batched = Qwen3TTSTokenizerV2Decoder._batch_dynamic_caches(caches)

    assert kv_ids.isdisjoint(copied_tensor_ids)
    assert batched is not caches[0]
    assert batched.layers is not caches[0].layers
    torch.testing.assert_close(batched.layers[0].keys, torch.cat(original_keys), rtol=0, atol=0)
    torch.testing.assert_close(batched.layers[0].values, torch.cat(original_values), rtol=0, atol=0)
    for index, layer in enumerate(batched.layers):
        source = caches[0].layers[index]
        assert type(layer) is type(source)
        assert layer is not source
        assert layer.is_initialized == source.is_initialized
        assert layer.get_seq_length() == source.get_seq_length()
        assert layer.sliding_window == source.sliding_window == 8
        torch.testing.assert_close(layer._sliding_window_tensor, source._sliding_window_tensor)
        assert layer._sliding_window_tensor.data_ptr() != source._sliding_window_tensor.data_ptr()
        layer._sliding_window_tensor.add_(1)
        assert source._sliding_window_tensor.item() == 8
    assert batched.layers[1].keys is None
    assert batched.layers[1].values is None
    for cache in caches:
        for name in ("keys", "values"):
            assert (
                getattr(batched.layers[0], name).untyped_storage().data_ptr()
                != getattr(cache.layers[0], name).untyped_storage().data_ptr()
            )
    batched.layers[0].keys.zero_()
    batched.layers[0].values.zero_()
    batched.layers[0].cumulative_length += 1
    for row, cache in enumerate(caches):
        torch.testing.assert_close(cache.layers[0].keys, original_keys[row], rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[0].values, original_values[row], rtol=0, atol=0)
        assert cache.get_seq_length() == 3


def test_batch_dynamic_caches_handles_shared_first_request_kv():
    caches = [DynamicCache(), DynamicCache()]
    keys = torch.ones(1, 2, 3, 4)
    caches[0].update(keys, keys, 0)
    caches[0].layers[0].values = caches[0].layers[0].keys
    caches[1].update(keys * 2, keys * 3, 0)

    batched = Qwen3TTSTokenizerV2Decoder._batch_dynamic_caches(caches)

    torch.testing.assert_close(batched.layers[0].keys, torch.cat([keys, keys * 2]))
    torch.testing.assert_close(batched.layers[0].values, torch.cat([keys, keys * 3]))
    assert batched.layers[0].keys.data_ptr() != batched.layers[0].values.data_ptr()


@pytest.mark.parametrize("initialized_field", [None, "keys", "values"])
def test_batch_dynamic_caches_rejects_mixed_initialization(initialized_field):
    config = Qwen3TTSTokenizerV2DecoderConfig(num_hidden_layers=1)
    caches = [DynamicCache(config=config), DynamicCache(config=config)]
    keys = torch.ones(1, 2, 3, 4)
    caches[0].update(keys, keys, 0)
    if initialized_field is not None:
        setattr(caches[1].layers[0], initialized_field, keys)

    with pytest.raises(ValueError, match="Cannot batch partially initialized request KV caches"):
        Qwen3TTSTokenizerV2Decoder._batch_dynamic_caches(caches)
