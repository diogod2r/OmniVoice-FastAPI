# OmniVoice FastAPI

Dockerized FastAPI wrapper for [OmniVoice](https://github.com/k2-fsa/OmniVoice) — zero-shot TTS for **600+ languages** with voice cloning and voice design. Fully **OpenAI-compatible** speech endpoint.

---

## Features

- **`/v1/audio/speech`** — OpenAI-compatible drop-in replacement (works with Open WebUI, SillyTavern, etc.)
- **`/v1/audio/clone`** — Zero-shot voice cloning from a reference audio file
- **`/v1/audio/design`** — Voice design via natural language attributes (e.g. `"female, low pitch, british accent"`)
- **`/web`** — Built-in web UI with TTS, Clone, and Design tabs
- **`/docs`** — Swagger UI with inline audio player for all endpoints

---

## Quick Start

### GPU (recommended)

```bash
git clone https://github.com/your-username/omnivoice-fastapi.git
cd omnivoice-fastapi/docker/gpu
docker compose up --build
```

> First run downloads the model (~4 GB) to a Docker volume. Subsequent starts are instant.

### CPU

```bash
cd omnivoice-fastapi/docker/cpu
docker compose up --build
```

The API is available at **http://localhost:8880**.

---

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/v1/audio/speech` | POST JSON | OpenAI-compatible TTS |
| `/v1/audio/clone` | POST multipart | Voice cloning from reference audio |
| `/v1/audio/design` | POST multipart | Voice design via text attributes |
| `/v1/voices` | GET | List voice presets |
| `/v1/languages` | GET | Common language IDs |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/web` | GET | Web UI |
| `/docs` | GET | Swagger UI (with inline audio player) |

---

## OpenAI-Compatible Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="omnivoice",
    voice="female",          # preset or any free-form instruct string
    input="Hello from OmniVoice!",
    extra_body={"language_id": "pt"},  # optional: force language
) as response:
    response.stream_to_file("output.wav")
```

---

## Voice Presets (`voice` field)

| ID | Description |
|----|-------------|
| `auto` | Model chooses automatically |
| `female` | Generic female |
| `male` | Generic male |
| `female_en` | Female, American accent |
| `male_en` | Male, American accent |
| `female_br` | Female, British accent |
| `male_br` | Male, British accent |
| `child` | Child voice |
| `elderly` | Elderly female |
| `whisper` | Whispering female |

You can also pass any **free-form instruct string** as the `voice` parameter:
```
voice="male, low pitch, australian accent"
```

---

## Voice Cloning (curl)

```bash
curl -X POST http://localhost:8880/v1/audio/clone \
  -F "text=Hello, I am speaking in your voice." \
  -F "ref_audio=@reference.wav" \
  -F "language_id=pt" \
  --output cloned.wav
```

> **Tip for Portuguese**: pass `language_id=pt` to avoid a European accent when synthesizing Brazilian Portuguese.

## Voice Design (curl)

```bash
curl -X POST http://localhost:8880/v1/audio/design \
  -F "text=This voice was designed from scratch." \
  -F "instruct=female, low pitch, british accent" \
  --output designed.wav
```

---

## Generation Parameters

All three endpoints expose the full OmniVoice `generate()` API:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language_id` | auto | ISO language code (`pt`, `en`, `zh`, …) — see `/v1/languages` |
| `speed` | 1.0 | Speed factor (ignored when `duration` is set) |
| `duration` | — | Fixed output duration in seconds (overrides `speed`) |
| `num_step` | 32 | Diffusion steps (lower = faster, higher = better quality) |
| `guidance_scale` | 2.0 | Classifier-free guidance scale |
| `denoise` | true | Prepend `<\|denoise\|>` token (speech only) |
| `t_shift` | 0.1 | Noise schedule time-step shift |
| `position_temperature` | 5.0 | Mask-position selection temperature |
| `class_temperature` | 0.0 | Token sampling temperature |
| `layer_penalty_factor` | 5.0 | Deeper codebook layer penalty |
| `preprocess_prompt` | true | Remove silences from reference audio (clone only) |
| `postprocess_output` | true | Remove silences from generated audio |
| `audio_chunk_duration` | 15.0 | Long-form chunk size (seconds) |
| `audio_chunk_threshold` | 30.0 | Long-form activation threshold (seconds) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace model ID |
| `DEVICE` | auto-detected | `cuda:0`, `cpu`, or `mps` |
| `HF_HOME` | `/app/models` | HuggingFace cache dir (mounted as Docker volume) |
| `HF_ENDPOINT` | — | Mirror URL, e.g. `https://hf-mirror.com` |

---

## Project Structure

```
omnivoice-fastapi/
├── main.py              # FastAPI app — all endpoints
├── index.html           # Web UI (served at /web)
├── docker/
│   ├── gpu/
│   │   ├── Dockerfile           # python:3.10-slim + omnivoice (GPU)
│   │   └── docker-compose.yml   # NVIDIA GPU deployment
│   └── cpu/
│       ├── Dockerfile           # python:3.11-slim + torch CPU
│       └── docker-compose.yml   # CPU deployment
└── README.md
```

---

## License

MIT
