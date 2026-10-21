# Spoken Transcribe

Password-protected Streamlit page: upload audio, pick a model, download the transcript.

- Default: ElevenLabs Scribe v2, speaker labels, no timestamps (`.txt`).
- Optional: OpenAI models (GPT Transcribe, Whisper, GPT-4o Diarize, GPT-4o Mini), paragraph timestamps, subtitles (`.srt`), word timestamps (`.json`), language, speaker count, vocabulary hints, filler-word removal.

Deploy on Streamlit Community Cloud with main file `app.py` and these secrets:

```toml
ELEVENLABS_API_KEY = "..."
APP_PASSWORD = "..."
OPENAI_API_KEY = "..."   # optional, only for OpenAI models
```
