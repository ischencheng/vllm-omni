# MiniCPM-o 4.5 Thinker prefix cache on turn MRv2: raw results

Image `vllm/vllm-openai:v0.32.0` plus `uv pip install .` of the tree. 1x H100 for serving runs, one server at a time. The driver is `scripts/pc_bench.py`; its arms and workloads are in its docstring and argparse. Summaries: `python3 scripts/summary.py <dir> [<dir>]`, `python3 scripts/analyze.py <dir>`.

| directory | tree | command |
|---|---|---|
| `head-off-on` | `5a10eb2f1` | `pc_bench.py --arms v2-off,v2-native --rounds 2 --workloads sys_c1,sys_c8,long_c1,text,miss_c1,miss_c8,chat_c4` |
| `ab-five-arms-1`, `ab-five-arms-2` | `603cd73a8` + `scripts/ab-five-arms-tree.patch` | `pc_bench.py --arms v1-off,v1-on,v2-off,v2-merge,v2-native` and the reverse order (older driver) |
| `correctness` | `5a10eb2f1` | `pc_bench.py --dump-rows --arms v2-native,v2-native-ds,v2-off-ds --workloads correct,correct_mm,stress` |
| `fp32` | `76b7ca2b7` | `pc_bench.py --dump-rows --arms v2-off-fp32,v2-native-fp32 --workloads correct` |
| `one-token` | `76b7ca2b7` | `pc_bench.py --arms v1-on,v2-off,v2-native --workloads edge,correct` |
| `low-kv-1.6g`, `low-kv-0.6g` | `76b7ca2b7` | `pc_bench.py --arms v2-native-tiny,v2-off-tiny --workloads stress`, `pc_bench.py --dump-rows --arms v2-native-tiny2 --workloads stress` |
| `main-cache-on` | main `01c3f711a` | `pc_bench.py --arms v2-native --workloads sys_c1,long_c1` |
| `main-one-token` | main `01c3f711a` | `pc_bench.py --arms v1-on,v2-off --workloads edge` |
| `duplex-e2e` | `5a10eb2f1` | `duplex_e2e.py merge` |
| `unit-head`, `unit-main-red` | `ed5c8c8e7`, main `01c3f711a` with the new tests | `unit.py <paths in the PR>` |
| `unit-head-handoff` | `b43158bf9` | `unit.py tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py` |

The saved rows (`*.pt`) stay out of this branch. `rows-compare.txt` compares them per prompt, first pass against second pass, and `rows-gap.txt` counts reply tokens minus rows after the 0.6 GiB run. In `correctness`, the `v2-off-fp32` and `v2-native-fp32` directories are start-up failures from a KV budget too small for FP32 weights; `fp32` is the run with `kv_cache_memory_bytes` set. In `low-kv-1.6g` the warm-up's long prompts return 400 because of `max_model_len`, which are the two tracebacks in those logs.
