# Raw results for JoyAI duplex streaming speech (`feat/joyai-streaming-speech`)

- Image: `vllm/vllm-openai:v0.32.0` (vLLM 0.32.0) plus `pip install .` of the revision named per directory.
- GPU runs: 1x NVIDIA H100 80GB HBM3, driver 610.57.04, one server at a time in one container. CPU runs: 8 vCPUs.
- Models: `jdopensource/JoyAI-VL-Interaction-Preview`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (TTS reference only), `openai/whisper-small.en` (speech check only).
- Client: the repo's `vllm_omni.clients.duplex.DuplexClient` + `EventCollector`, monotonic clock. Same flow as `tests/e2e/online_serving/test_joyai_vl_interaction.py`: a user text item ("describe each new image aloud in two or three sentences"), then one `append_audio(b"", video_frames=[jpeg])` + `commit(create_response=True)` per image, `temperature=0`, `tts_language=English`, playback ack after the next commit. A red warm-up image is first and is not counted. A second session is told to speak only on fire or smoke and gets two ordinary images.
- Images: vLLM's `stop_sign` and `cherry_blossom` assets, plus two drawn in `scripts/measure.py` (`drawn_house`, `drawn_whiteboard`).

| dir | what | revision | command |
| --- | --- | --- | --- |
| `latency/` | default duplex deploy (`a`) vs `joyai_vl_interaction_duplex_streaming.yaml` (`c`) | `274d2efb5` | `VLLM_OMNI_TREE=<tree> OUT_DIR=<out> python scripts/measure.py --arms a,c`, then `python scripts/continuity.py <tree> latency/events_joyai_duplex_streaming.json latency/events_joyai_duplex.json` |
| `chunk-variants/` | streaming deploy with a 1-frame first chunk (`c`), `codec_chunk_ramp: [4, 4, 8, 16, 25]` (`r`), `initial_codec_chunk_frames: 4` (`i4`) | `22a3804d4` (before the ramp went into the overlay) | `python scripts/measure.py --arms c,r,i4`, then `scripts/continuity.py` on the three `events_*.json` |
| `upstream-and-tts-reference/` | upstream default duplex deploy (`a`); Qwen3-TTS alone on `/v1/audio/speech` (`stream_format=audio`, `pcm`, `Vivian`) for the same texts, with `model_runner: v1` and the deploy default | `8d4b2789a` (upstream main) | `python scripts/measure.py --arms a,b` |
| `speech-check/` | Whisper transcripts of every `latency/` reply against JoyAI's text | — | WAVs rebuilt from `latency/events_*.json`, then `python scripts/asr.py` |
| `e2e/` | `pytest -v --run-level=advanced_model tests/e2e/online_serving/test_joyai_vl_interaction.py tests/e2e/online_serving/test_joyai_vl_delegation.py`, Hub online | `22a3804d4` | `python scripts/e2e.py --online` |
| `unit/` | `pytest -m cpu tests/model_executor/models/joyai_vl_interaction tests/model_executor/stage_input_processors/test_joyai_vl_interaction.py tests/config/test_config_merge.py` | `274d2efb5` | `python scripts/unit.py` |

`274d2efb5` only adds the chunk ramp to the streaming overlay and a test; `f1dde13eb` only edits the recipe. Neither touches the default profiles that `e2e/` runs.

In each run directory: `summary.json` has one row per reply (`first_transcript_ms`, `first_audio_ms`, `done_ms`, `audio_delta_events`, `audio_s`, `text`), plus the silence-session rows; `events_*.json` is every client event with `_client_received_at_s` and the base64 audio; `server_*.log` is the server log; `driver.log` is the measurement log; `continuity.jsonl` is `compute_continuity_stats` per reply (100 ms budget).

Volume paths in the logs are rewritten to `/cache`.

## Numbers in the PR, and where they come from

- First audio, done, audio deltas: `latency/summary.json` `arm_a` / `arm_c`, rows 1-4 (row 0 is the warm-up).
- Underruns: `latency/continuity.jsonl` and `chunk-variants/continuity.jsonl`.
- Silent observations: the `*_silence` rows in `latency/summary.json` and `chunk-variants/summary.json`.
- WER: `speech-check/asr.json`.
- TTS-only first packet: `upstream-and-tts-reference/summary.json` `arm_b`.
- `e2e/e2e.log`: `2 passed`. `unit/unit.log`: `322 passed`.
