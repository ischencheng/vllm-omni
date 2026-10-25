"""Moshi parity probe (measurement only).

With MOSHI_TRACE_DIR set, wraps PersonaPlexTalkerForConditionalGeneration.post_sample_talker_mtp
(the eager depformer path; run with depformer_cuda_graphs off) and appends one JSON line
per call to <MOSHI_TRACE_DIR>/<pid>.jsonl: wall time, request ids, the sampled text
token and the raw depformer codes of every row.
"""

import json
import os
import sys
import threading
import time

TRACE_DIR = os.environ.get("MOSHI_TRACE_DIR")

if TRACE_DIR:
    import functools
    import importlib.abc

    _lock = threading.Lock()
    _fh = None

    def _emit(**fields):
        global _fh
        fields["pid"] = os.getpid()
        with _lock:
            if _fh is None:
                os.makedirs(TRACE_DIR, exist_ok=True)
                _fh = open(os.path.join(TRACE_DIR, f"{os.getpid()}.jsonl"), "a", buffering=1)
            _fh.write(json.dumps(fields, default=str) + "\n")

    def _patch_talker(mod):
        cls = mod.PersonaPlexTalkerForConditionalGeneration
        orig = cls.post_sample_talker_mtp

        @functools.wraps(orig)
        def post_sample_talker_mtp(self, *, input_ids, hidden_states, req_ids, req_infos):
            t0 = time.time()
            codes = orig(self, input_ids=input_ids, hidden_states=hidden_states, req_ids=req_ids, req_infos=req_infos)
            try:
                _emit(ev="post_sample", t=t0, req_ids=list(req_ids),
                      text=input_ids.reshape(-1).tolist(), codes=codes.reshape(len(req_ids), -1).tolist(),
                      graphs=getattr(self, "_depformer_graphs", None) is not None)
            except Exception as e:  # keep serving
                _emit(ev="probe_error", err=repr(e))
            return codes

        cls.post_sample_talker_mtp = post_sample_talker_mtp

    def _patch_stage0(mod):
        cls = mod.PersonaPlexStage0DuplexRuntime
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            _emit(ev="runtime", moshi=bool(getattr(self, "moshi", False)), model_path=kwargs.get("model_path"))

        cls.__init__ = __init__

    def _patch_runner(mod):
        runner = mod.GPUARModelRunner
        for meth in ("execute_model", "sample_tokens"):
            orig = getattr(runner, meth)

            def make(orig, meth):
                @functools.wraps(orig)
                def wrapped(self, *args, **kwargs):
                    t0 = time.time()
                    p0 = time.perf_counter()
                    r = orig(self, *args, **kwargs)
                    so = args[0] if args else kwargs.get("scheduler_output")
                    nreq = None
                    try:
                        nreq = len(so.num_scheduled_tokens)
                    except Exception:
                        pass
                    _emit(ev="step", m=meth, t=t0, dur_ms=(time.perf_counter() - p0) * 1e3, nreq=nreq)
                    return r
                return wrapped

            setattr(runner, meth, make(orig, meth))

    PATCHES = {
        "vllm_omni.worker.gpu_ar_model_runner": _patch_runner,
        "vllm_omni.model_executor.models.personaplex.personaplex_talker": _patch_talker,
        "vllm_omni.model_executor.models.personaplex.duplex.stage0": _patch_stage0,
    }

    class _PostImportFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = PATCHES.get(fullname)
            if patch is None:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    patch(module)
                    _emit(ev="patched", module=fullname)
                except Exception as e:
                    _emit(ev="patch_error", module=fullname, err=repr(e))

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PostImportFinder())
