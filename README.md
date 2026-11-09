# Nemotron 3.5 ASR Streaming — Local Web Demo

**English** | [简体中文](README.zh-CN.md)

A local web demo for [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), running fully offline on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). **True streaming** live captions from your microphone + batch file transcription, with a self-implemented **hotword biasing** feature on top.

- Model: 600M-parameter cache-aware FastConformer-RNNT, one checkpoint covering 40 language-locales (incl. Chinese), with punctuation & capitalization
- Weights: [mlx-community bf16 conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) (~1.2 GB), auto-downloaded on first run
- Measured on an M5 Max: batch RTFx 15–20, streaming RTFx 4–7 (hotword biasing included), solid for English and Chinese

## Requirements

- Apple Silicon Mac (M1 or later) — MLX is Apple-Silicon-only
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A modern browser (Chrome / Safari / Edge) for microphone capture

## Install & Run

```bash
git clone https://github.com/misakaikato/nemotron-asr-demo.git
cd nemotron-asr-demo
uv run server.py
```

That's it. `uv run` creates the virtualenv and installs all locked dependencies on first invocation (note: `mlx-audio` is installed from GitHub `main`, since Nemotron support hasn't shipped to PyPI yet). Model weights are fetched from Hugging Face on first start (~1.2 GB, cached afterwards).

Open **http://localhost:7860** and grant microphone permission when prompted.

To keep the project fully offline / self-contained, place the weights inside the project instead — the server prefers a local copy when present:

```bash
hf download mlx-community/nemotron-3.5-asr-streaming-0.6b \
  --local-dir models/nemotron-3.5-asr-streaming-0.6b
```

### Deploy notes

- The server binds to `127.0.0.1:7860` (see the last line of `server.py`). To reach it from other devices on your LAN, change the host to `0.0.0.0` — but note browsers only allow microphone access on `localhost` or HTTPS origins, so remote clients will need a TLS reverse proxy (e.g. Caddy) for the mic path; file transcription works over plain HTTP.
- Single-user demo: no auth, no concurrency control. Don't expose it to the public internet as-is.

## Usage

**Live captions** — pick a language (or leave on auto-detect), pick a latency tier, hit *Start recording* and talk. Text streams in as you speak; the status line shows audio duration / compute time / RTFx (mic waiting time excluded; >1 means faster than real time).

- **Latency tier** = the model's trained look-ahead (`att_context`): 1.12s / 0.56s / 0.32s / 0.08s. Smaller = snappier captions, slightly lower accuracy.
- **Hotwords** — one word or phrase per line, filled in **before** starting a session. Default weight 3 is a good start; too high (>6) will bend similar-sounding words into hotwords. Greedy decoding has no backtracking, so keep the list small (dozens, not thousands).
- **File transcription** — upload wav/flac/mp3 at the bottom for a pure-speed benchmark. Two synthesized samples live in `samples/`.

Verify hotword biasing from the CLI (the baseline mishears "上证指数" as "上政指数" in `samples/test-zh.wav`):

```bash
curl -X POST "http://127.0.0.1:7860/transcribe?language=auto" \
  -F "file=@samples/test-zh.wav" -F "hotwords=上证指数" -F "hotword_weight=3"
```

### API

| Endpoint | Protocol | Description |
|---|---|---|
| `/ws` | WebSocket | First message: JSON `{language, latency, hotwords, hotword_weight}`. Then send 16 kHz mono int16 PCM as binary frames; send the text `stop` to finish. Server pushes cumulative `{type:"partial", text, audio_s, compute_s, rtfx}` and a final `{type:"final"}`. |
| `/transcribe` | POST multipart | Fields: `file` (audio), `hotwords` (newline-separated, optional), `hotword_weight` (optional, default 3). Query: `language` (default `auto`). Returns `{text, audio_s, compute_s, rtfx}`. |

## Implementation notes

- **True streaming, not re-transcription**: reuses mlx-audio's internal pipeline — `log_mel_spectrogram_frames` for incremental mel extraction (frame-identical to a full-audio pass) → `stream_encode_chunks` (per-layer attention/conv caches, O(n), no recompute) → greedy RNNT decoding frame by frame. Mic PCM flows over the WebSocket and is fed to the encoder every ~0.32s of accumulated mel frames.
- **Hotword biasing** (not in mlx-audio; implemented here): shallow fusion in *text space*. mlx-audio's tokenizer only decodes, so instead of tokenizing hotwords, we pre-scan the vocabulary once per hotword position to build a table of "which tokens advance this match by how many characters". At each decode step, matching tokens get a logit bonus of `weight × chars_advanced`; emitted tokens advance or kill match states. This is segmentation-agnostic — it works for Chinese, English, and multi-word phrases regardless of how the model happens to split them. NVIDIA's official route (NeMo GPU-PB / [TurboBias](https://arxiv.org/abs/2508.07014)) is CUDA-only.
- `decode_biased` in `server.py` mirrors the greedy loop of `model._decode_prompted_chunks` with the bias injected — if you upgrade mlx-audio and that upstream function changes, re-align this copy.

## Project layout

```
server.py     # FastAPI: /ws streaming, /transcribe batch, HotwordBiaser + decode_biased
index.html    # single-page frontend: mic capture (ScriptProcessor → 16k PCM), hotwords, metrics
samples/      # test-en.wav / test-zh.wav (synthesized with macOS `say`, for verification)
models/       # (gitignored) optional local weights, see Install & Run
```

## Known limitations

- ScriptProcessor is deprecated but dependency-free; swap for AudioWorklet if you hit glitches
- Hotwords are not persisted (cleared on refresh) and don't affect an in-flight session
- Single-user demo: one decode thread per WebSocket session, no queueing

## License

Demo code: MIT. Model weights are distributed by NVIDIA under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (the mlx-community conversion inherits it) — review it before commercial use.

---

Contact · 联系方式：[misakaikato@outlook.com](mailto:misakaikato@outlook.com)
