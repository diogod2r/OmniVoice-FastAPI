import io
import os
import base64
import time
import logging
import tempfile
from typing import Optional, Literal
from contextlib import asynccontextmanager

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model ──────────────────────────────────────────────────────────────
omnivoice_model = None


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    global omnivoice_model
    from omnivoice import OmniVoice

    model_id = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
    device   = os.getenv("DEVICE", get_device())
    dtype    = torch.float16 if device != "cpu" else torch.float32

    logger.info(f"Loading OmniVoice model '{model_id}' on {device} ({dtype})...")
    omnivoice_model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
    logger.info("Model loaded ✓")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OmniVoice FastAPI",
    description=(
        "OpenAI-compatible TTS API powered by [OmniVoice](https://github.com/k2-fsa/OmniVoice) — "
        "600+ languages, voice cloning & voice design.\n\n"
        "**Endpoints**\n"
        "- `/v1/audio/speech` — Auto / instruct voice (OpenAI-compatible drop-in)\n"
        "- `/v1/audio/clone` — Zero-shot voice cloning from reference audio\n"
        "- `/v1/audio/design` — Voice design via natural language attributes\n\n"
        "**Language tip**: pass `language_id='pt'` for Portuguese to avoid European accent. "
        "Full list at `GET /v1/languages`."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Types ─────────────────────────────────────────────────────────────────────
AudioFormat = Literal["wav", "mp3", "flac", "ogg", "opus"]

# ── Schemas ───────────────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    # OpenAI-compatible fields
    model: str = Field("omnivoice", description="Model name (ignored)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        "auto",
        description=(
            "Preset or free-form instruct string. "
            "Presets: auto, female, male, female_en, male_en, female_br, male_br, child, elderly, whisper. "
            "Or any string like 'female, low pitch, british accent'. See GET /v1/voices."
        ),
    )
    response_format: AudioFormat = Field("wav", description="Output audio format")

    # Language
    language_id: Optional[str] = Field(
        None,
        description="ISO language ID (e.g. 'pt', 'en', 'zh'). Auto-detected from text if omitted. See GET /v1/languages.",
    )

    # Duration & Speed (priority: duration > speed)
    speed: Optional[float] = Field(
        None, ge=0.1, le=10.0,
        description="Speed factor (>1.0 faster, <1.0 slower). Ignored when duration is set. Default: 1.0.",
    )
    duration: Optional[float] = Field(
        None, ge=0.1,
        description="Fixed output duration in seconds. Overrides speed when set.",
    )

    # Decoding
    num_step: int = Field(32, ge=1, le=128, description="Diffusion steps. Higher = better quality & slower. Use 16 for fast inference.")
    guidance_scale: float = Field(2.0, ge=0.0, le=20.0, description="Classifier-free guidance scale.")
    denoise: bool = Field(True, description="Prepend <|denoise|> token for cleaner speech.")
    t_shift: float = Field(0.1, description="Time-step shift for the noise schedule.")

    # Sampling
    position_temperature: float = Field(5.0, description="Temperature for mask-position selection. 0 = greedy/deterministic.")
    class_temperature: float = Field(0.0, description="Temperature for token sampling at each step. 0 = greedy/deterministic.")
    layer_penalty_factor: float = Field(5.0, description="Penalty for deeper codebook layers (encourages lower layers to unmask first).")

    # Post-processing
    postprocess_output: bool = Field(True, description="Remove long silences from generated audio.")

    # Long-form generation
    audio_chunk_duration: float = Field(15.0, description="Target chunk duration (s) for long-form text splitting.")
    audio_chunk_threshold: float = Field(30.0, description="Estimated duration (s) above which chunking is activated.")


# ── Helpers ───────────────────────────────────────────────────────────────────

VOICE_PRESETS: dict = {
    "auto":      None,
    "female":    "female",
    "male":      "male",
    "female_en": "female, american accent",
    "male_en":   "male, american accent",
    "female_br": "female, british accent",
    "male_br":   "male, british accent",
    "child":     "child",
    "elderly":   "elderly, female",
    "whisper":   "female, whisper",
}


def resolve_instruct(voice: str) -> Optional[str]:
    if voice in VOICE_PRESETS:
        return VOICE_PRESETS[voice]
    return voice if voice and voice != "auto" else None


def audio_to_bytes(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Convert np.ndarray (T,) at sample_rate Hz to bytes.

    OmniVoice returns float32 arrays. soundfile defaults to 32-bit float WAV
    which most browsers cannot play in an <audio> element. Force PCM_16 so
    every major browser (Chrome, Firefox, Safari) can play the file inline.
    """
    buf = io.BytesIO()
    if fmt == "flac":
        sf.write(buf, audio, sample_rate, format="FLAC")
    elif fmt == "ogg":
        sf.write(buf, audio, sample_rate, format="OGG", subtype="VORBIS")
    elif fmt in ("mp3", "opus"):
        logger.warning(f"Format '{fmt}' not supported by soundfile, falling back to wav")
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    else:
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


MIME = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "flac": "audio/flac",
    "ogg":  "audio/ogg",
    "opus": "audio/opus",
}


# ── OpenAPI response schema shared by all audio endpoints ───────────────────
_AUDIO_RESPONSES = {
    200: {
        "content": {
            "audio/wav":  {"schema": {"type": "string", "format": "binary"}},
            "audio/flac": {"schema": {"type": "string", "format": "binary"}},
            "audio/ogg":  {"schema": {"type": "string", "format": "binary"}},
        },
        "description": "Audio file. Swagger UI renders an inline player for WAV responses.",
    }
}


# ── /v1/audio/speech — OpenAI-compatible TTS ──────────────────────────────────

@app.post("/v1/audio/speech", tags=["OpenAI Compatible"], response_class=Response, responses=_AUDIO_RESPONSES)
async def openai_tts(req: SpeechRequest):
    """
    OpenAI-compatible speech synthesis. Drop-in replacement: set `base_url="http://host:8880/v1"`.

    **Voice modes** (via `voice` field):
    - Preset: `auto`, `female`, `male`, `female_en`, `male_en`, `female_br`, `male_br`, `child`, `elderly`, `whisper`
    - Free-form instruct string: `"female, low pitch, british accent"`
    - `"auto"` or omit: model picks a voice automatically

    **Priority**: `duration` > `speed`.
    """
    if not omnivoice_model:
        raise HTTPException(503, "Model not loaded")

    instruct = resolve_instruct(req.voice)
    t0 = time.time()
    try:
        kwargs: dict = {
            "text":                  req.input,
            "num_step":              req.num_step,
            "guidance_scale":        req.guidance_scale,
            "denoise":               req.denoise,
            "t_shift":               req.t_shift,
            "position_temperature":  req.position_temperature,
            "class_temperature":     req.class_temperature,
            "layer_penalty_factor":  req.layer_penalty_factor,
            "postprocess_output":    req.postprocess_output,
            "audio_chunk_duration":  req.audio_chunk_duration,
            "audio_chunk_threshold": req.audio_chunk_threshold,
        }
        if instruct:
            kwargs["instruct"] = instruct
        if req.language_id:
            kwargs["language_id"] = req.language_id
        if req.duration is not None:
            kwargs["duration"] = req.duration
        elif req.speed is not None:
            kwargs["speed"] = req.speed

        audio_list = omnivoice_model.generate(**kwargs)
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(500, str(e))

    audio = audio_list[0]
    elapsed = time.time() - t0
    duration_s = len(audio) / 24000
    logger.info(f"TTS: {duration_s:.2f}s audio in {elapsed:.2f}s (RTF={elapsed/duration_s:.3f})")

    data = audio_to_bytes(audio, 24000, req.response_format)
    return Response(
        content=data,
        media_type=MIME.get(req.response_format, "audio/wav"),
        headers={"Content-Length": str(len(data)), "X-RTF": str(round(elapsed / duration_s, 4))},
    )


# ── /v1/audio/clone — Voice Cloning ──────────────────────────────────────────

@app.post("/v1/audio/clone", tags=["Voice Cloning"], response_class=Response, responses=_AUDIO_RESPONSES)
async def clone_voice(
    text: str = Form(..., description="Text to synthesize in the cloned voice"),
    ref_audio: Optional[UploadFile] = File(None, description="Reference audio file (WAV/MP3/M4A). 3–10 seconds recommended. Use this OR ref_audio_base64."),
    ref_audio_base64: Optional[str] = Form(None, description="Reference audio as base64-encoded string (WAV/MP3/M4A). Use this OR ref_audio file upload."),
    ref_text: Optional[str] = Form(
        None,
        description="Transcription of ref_audio. If omitted, Whisper ASR auto-transcribes it.",
    ),
    language_id: Optional[str] = Form(
        None,
        description="Language ID (e.g. 'pt' for Portuguese, 'en' for English). Auto-detected if omitted. See GET /v1/languages.",
    ),
    speed: Optional[float] = Form(None, description="Speed factor (>1 faster, <1 slower). Ignored when duration is set."),
    duration: Optional[float] = Form(None, description="Fixed output duration in seconds. Overrides speed."),
    num_step: int = Form(32, description="Diffusion steps (1–128). Use 16 for faster inference."),
    guidance_scale: float = Form(2.0, description="Classifier-free guidance scale."),
    preprocess_prompt: bool = Form(True, description="Preprocess reference audio (remove long silences, add punctuation to ref_text)."),
    postprocess_output: bool = Form(True, description="Remove long silences from generated audio."),
    audio_chunk_duration: float = Form(15.0, description="Target chunk duration (s) for long-form generation."),
    audio_chunk_threshold: float = Form(30.0, description="Estimated duration (s) above which chunking is activated."),
    response_format: AudioFormat = Form("wav", description="Output audio format"),
):
    """
    Zero-shot voice cloning from a reference audio file.

    Upload any WAV/MP3/M4A reference (3–10s recommended) and the text to synthesize.

    **Tips**:
    - Set `language_id='pt'` when synthesizing Portuguese to avoid European accent.
    - Leave `ref_text` blank to auto-transcribe via Whisper (slower on first call).
    - Use a 3–10s clean reference clip for best cloning quality.

    **Priority**: `duration` > `speed`.
    """
    if not omnivoice_model:
        raise HTTPException(503, "Model not loaded")

    if not ref_audio and not ref_audio_base64:
        raise HTTPException(400, "Provide ref_audio (file upload) or ref_audio_base64 (base64 string)")

    if ref_audio_base64:
        try:
            # Strip data URI prefix if present (e.g. "data:audio/webm;base64,...")
            if "," in ref_audio_base64:
                ref_audio_base64 = ref_audio_base64.split(",", 1)[1]
            audio_bytes = base64.b64decode(ref_audio_base64)
        except Exception:
            raise HTTPException(400, "ref_audio_base64 is not valid base64")
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
    else:
        suffix = os.path.splitext(ref_audio.filename or "ref.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await ref_audio.read())
            tmp_path = tmp.name

    try:
        kwargs: dict = {
            "text":                  text,
            "ref_audio":             tmp_path,
            "num_step":              num_step,
            "guidance_scale":        guidance_scale,
            "preprocess_prompt":     preprocess_prompt,
            "postprocess_output":    postprocess_output,
            "audio_chunk_duration":  audio_chunk_duration,
            "audio_chunk_threshold": audio_chunk_threshold,
        }
        if ref_text:
            kwargs["ref_text"] = ref_text
        if language_id:
            kwargs["language_id"] = language_id
        if duration is not None:
            kwargs["duration"] = duration
        elif speed is not None:
            kwargs["speed"] = speed

        audio_list = omnivoice_model.generate(**kwargs)
    except Exception as e:
        logger.exception("Clone generation failed")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    audio = audio_list[0]
    logger.info(f"Clone: generated {len(audio)/24000:.2f}s audio")
    data = audio_to_bytes(audio, 24000, response_format)
    return Response(
        content=data,
        media_type=MIME.get(response_format, "audio/wav"),
        headers={"Content-Length": str(len(data))},
    )


# ── /v1/audio/design — Voice Design ──────────────────────────────────────────

@app.post("/v1/audio/design", tags=["Voice Design"], response_class=Response, responses=_AUDIO_RESPONSES)
async def design_voice(
    text: str = Form(..., description="Text to synthesize"),
    instruct: str = Form(
        ...,
        description=(
            "Voice attributes (comma-separated). "
            "Supported: gender (male/female), age (child/elderly), "
            "pitch (very low/low/high/very high), style (whisper), "
            "English accents (American, British, Australian…), "
            "Chinese dialects (四川话, 陕西话…). "
            "Example: 'female, low pitch, british accent'"
        ),
    ),
    language_id: Optional[str] = Form(
        None,
        description="Language ID (e.g. 'pt'). Auto-detected if omitted. Note: Voice Design was trained mainly on Chinese and English data.",
    ),
    speed: Optional[float] = Form(None, description="Speed factor. Ignored when duration is set."),
    duration: Optional[float] = Form(None, description="Fixed output duration in seconds. Overrides speed."),
    num_step: int = Form(32, description="Diffusion steps (1–128). Use 16 for faster inference."),
    guidance_scale: float = Form(2.0, description="Classifier-free guidance scale."),
    postprocess_output: bool = Form(True, description="Remove long silences from generated audio."),
    audio_chunk_duration: float = Form(15.0, description="Target chunk duration (s) for long-form generation."),
    audio_chunk_threshold: float = Form(30.0, description="Estimated duration (s) above which chunking is activated."),
    response_format: AudioFormat = Form("wav", description="Output audio format"),
):
    """
    Design a voice using natural language attribute descriptions — no reference audio needed.

    **Note**: Voice Design was trained on Chinese and English data.
    It generalises to other languages but may produce unstable results for low-resource languages.

    **Priority**: `duration` > `speed`.
    """
    if not omnivoice_model:
        raise HTTPException(503, "Model not loaded")

    try:
        kwargs: dict = {
            "text":                  text,
            "instruct":              instruct,
            "num_step":              num_step,
            "guidance_scale":        guidance_scale,
            "postprocess_output":    postprocess_output,
            "audio_chunk_duration":  audio_chunk_duration,
            "audio_chunk_threshold": audio_chunk_threshold,
        }
        if language_id:
            kwargs["language_id"] = language_id
        if duration is not None:
            kwargs["duration"] = duration
        elif speed is not None:
            kwargs["speed"] = speed

        audio_list = omnivoice_model.generate(**kwargs)
    except Exception as e:
        logger.exception("Design generation failed")
        raise HTTPException(500, str(e))

    audio = audio_list[0]
    logger.info(f"Design: generated {len(audio)/24000:.2f}s audio")
    data = audio_to_bytes(audio, 24000, response_format)
    return Response(
        content=data,
        media_type=MIME.get(response_format, "audio/wav"),
        headers={"Content-Length": str(len(data))},
    )


# ── Info endpoints ────────────────────────────────────────────────────────────

@app.get("/v1/languages", tags=["Info"])
async def list_languages():
    """
    Common language IDs to pass as `language_id` in any generation endpoint.

    Full list (646 languages + training hours):
    https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md
    """
    return {
        "note": "Pass language_id to any generation endpoint to force a specific language/accent.",
        "common": [
            {"id": "pt",  "name": "Portuguese (Brazil & Portugal)", "training_hours": 16855},
            {"id": "en",  "name": "English",                        "training_hours": 206061},
            {"id": "es",  "name": "Spanish",                        "training_hours": 27559},
            {"id": "fr",  "name": "French",                         "training_hours": 23675},
            {"id": "de",  "name": "German",                         "training_hours": 21927},
            {"id": "it",  "name": "Italian",                        "training_hours": 9402},
            {"id": "ja",  "name": "Japanese",                       "training_hours": 36914},
            {"id": "ko",  "name": "Korean",                         "training_hours": 8609},
            {"id": "zh",  "name": "Chinese (Mandarin)",             "training_hours": 111343},
            {"id": "ru",  "name": "Russian",                        "training_hours": 20338},
            {"id": "arb", "name": "Arabic (Standard)",              "training_hours": 1483},
            {"id": "hi",  "name": "Hindi",                          "training_hours": 117},
        ],
        "full_list_url": "https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md",
    }


@app.get("/v1/models", tags=["Info"])
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{"id": "omnivoice", "object": "model", "owned_by": "k2-fsa"}],
    }


@app.get("/v1/voices", tags=["Info"])
async def list_voices():
    """List voice presets for the `voice` field in /v1/audio/speech."""
    return {
        "voices": [
            {"id": "auto",      "description": "Auto-selected voice"},
            {"id": "female",    "description": "Female voice"},
            {"id": "male",      "description": "Male voice"},
            {"id": "female_en", "description": "Female, American English accent"},
            {"id": "male_en",   "description": "Male, American English accent"},
            {"id": "female_br", "description": "Female, British accent"},
            {"id": "male_br",   "description": "Male, British accent"},
            {"id": "child",     "description": "Child voice"},
            {"id": "elderly",   "description": "Elderly female voice"},
            {"id": "whisper",   "description": "Whispering female voice"},
        ],
        "note": "You can also pass any free-form instruct string as voice, e.g. 'female, low pitch, british accent'",
    }


@app.get("/health", tags=["Info"])
async def health():
    """Health check — reports model load status and device."""
    return {
        "status": "ok",
        "model_loaded": omnivoice_model is not None,
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
    }


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    web_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
    with open(web_path) as f:
        return f.read()
