"""
Transcription models offered on the page, grouped by provider.
Prices are batch (file upload) API rates per hour of audio, Oct 2026.
"""

ELEVENLABS = "elevenlabs"
OPENAI = "openai"

PROVIDER_NAMES = {ELEVENLABS: "ElevenLabs", OPENAI: "OpenAI"}
SECRET_NAMES = {ELEVENLABS: "ELEVENLABS_API_KEY", OPENAI: "OPENAI_API_KEY"}

# Keyterm prompting is an ElevenLabs add-on billed per audio hour.
ELEVENLABS_KEYTERMS_USD_PER_HOUR = 0.05

OPENAI_RETIREMENT = "retires Feb 26, 2027"

MODELS = [
    {
        "id": "scribe_v2",
        "provider": ELEVENLABS,
        "name": "Scribe v2",
        "usd_per_hour": 0.22,
        "speakers": True,
        "timed": True,        # paragraph timestamps + subtitles
        "word_timestamps": True,
        "deprecated": False,
        "why": "Best for multi-speaker and mixed-language audio. Speaker labels (up to 32), "
               "word timestamps, 90+ languages, filler-word removal. Files up to 5 GB in one go.",
    },
    {
        "id": "scribe_v1",
        "provider": ELEVENLABS,
        "name": "Scribe v1",
        "usd_per_hour": 0.22,
        "speakers": True,
        "timed": True,
        "word_timestamps": True,
        "deprecated": True,
        "why": "Older ElevenLabs model, same price, less accurate — replaced by v2. Only useful for comparison.",
    },
    {
        "id": "gpt-transcribe",
        "provider": OPENAI,
        "name": "GPT Transcribe",
        "usd_per_hour": 0.27,
        "speakers": False,
        "timed": False,
        "word_timestamps": False,
        "deprecated": False,
        "why": "OpenAI's newest and most accurate model. Text only — no speaker labels, no timestamps. "
               "Good for one speaker, and for vocabulary hints.",
    },
    {
        "id": "gpt-4o-transcribe-diarize",
        "provider": OPENAI,
        "name": "GPT-4o Transcribe Diarize",
        "usd_per_hour": 0.36,
        "speakers": True,
        "timed": True,
        "word_timestamps": False,
        "deprecated": True,
        "why": "OpenAI with speaker labels and per-sentence timestamps. On recordings longer than "
               "10 minutes, speaker numbers may not stay consistent. Ignores vocabulary hints.",
    },
    {
        "id": "whisper-1",
        "provider": OPENAI,
        "name": "Whisper",
        "usd_per_hour": 0.36,
        "speakers": False,
        "timed": True,
        "word_timestamps": True,
        "deprecated": True,
        "why": "Classic Whisper (large-v2). Word timestamps, no speaker labels. Weak when speakers mix languages.",
    },
    {
        "id": "gpt-4o-mini-transcribe",
        "provider": OPENAI,
        "name": "GPT-4o Mini Transcribe",
        "usd_per_hour": 0.18,
        "speakers": False,
        "timed": False,
        "word_timestamps": False,
        "deprecated": True,
        "why": "Cheapest. Text only, lower accuracy than GPT Transcribe.",
    },
]

MODELS_BY_ID = {m["id"]: m for m in MODELS}
DEFAULT_MODEL_ID = "scribe_v2"

LANGUAGES = {
    "auto": "Auto-detect (best for mixed languages)",
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "es": "Spanish",
}


def model_label(model_id: str) -> str:
    m = MODELS_BY_ID[model_id]
    label = f"{PROVIDER_NAMES[m['provider']]} · {m['name']} · ${m['usd_per_hour']:.2f}/h"
    if model_id == DEFAULT_MODEL_ID:
        label += " · default"
    elif m["deprecated"]:
        label += " · old"
    return label


def estimate_cost(model_id: str, duration_sec: float, keyterms: bool) -> float:
    m = MODELS_BY_ID[model_id]
    rate = m["usd_per_hour"]
    if keyterms and m["provider"] == ELEVENLABS and model_id == "scribe_v2":
        rate += ELEVENLABS_KEYTERMS_USD_PER_HOUR
    return duration_sec / 3600 * rate
