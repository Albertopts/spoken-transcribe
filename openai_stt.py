"""
OpenAI transcription → shared result shape (see formats.py).

OpenAI accepts at most 25 MB per request, so audio is converted to mono
16 kHz MP3 and split into ~10-minute pieces, cut at the quietest moment near
each boundary so words are not chopped. Timestamps are shifted back by each
piece's offset.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydub import AudioSegment

from formats import merge_same_speaker, paragraphs_from_cues, smart_join, speaker_numberer

CHUNK_MS = 10 * 60 * 1000
SEARCH_MS = 15 * 1000        # look for a quiet cut point ±15 s around each boundary
MIN_TAIL_MS = 30 * 1000      # a shorter remainder is merged into the last piece
WINDOW_MS = 100
EXPORT_BITRATE = "48k"       # 10 min ≈ 3.6 MB, far under the 25 MB limit
REQUEST_TIMEOUT_SEC = 900


def _quietest_point(audio: AudioSegment, start_ms: int, end_ms: int) -> int:
    start_ms = max(start_ms, 0)
    end_ms = min(end_ms, len(audio) - WINDOW_MS)
    if end_ms <= start_ms:
        return start_ms
    return min(range(start_ms, end_ms, WINDOW_MS), key=lambda t: audio[t:t + WINDOW_MS].rms)


def split_audio(filename: str, data: bytes, chunk_ms: int = CHUNK_MS) -> tuple[list[tuple[bytes, float]], float]:
    """Return ([(mp3_bytes, offset_seconds), ...], duration_seconds)."""
    suffix = Path(filename).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        audio = AudioSegment.from_file(tmp_path)  # a file path lets ffmpeg seek (needed for m4a)
    finally:
        os.unlink(tmp_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    cuts = [0]
    pos = chunk_ms
    while pos < len(audio) - MIN_TAIL_MS:
        cut = _quietest_point(audio, pos - SEARCH_MS, pos + SEARCH_MS)
        cuts.append(cut)
        pos = cut + chunk_ms
    cuts.append(len(audio))

    chunks = []
    for a, b in zip(cuts, cuts[1:]):
        out = tempfile.SpooledTemporaryFile()
        audio[a:b].export(out, format="mp3", bitrate=EXPORT_BITRATE)
        out.seek(0)
        chunks.append((out.read(), a / 1000.0))
    return chunks, len(audio) / 1000.0


def _as_dict(response) -> dict:
    if isinstance(response, str):
        return {"text": response}
    return response.model_dump() if hasattr(response, "model_dump") else dict(response)


def transcribe(
    api_key: str,
    filename: str,
    data: bytes,
    model_id: str,
    language: Optional[str] = None,
    keyterms: tuple = (),
    progress=None,
    chunk_ms: int = CHUNK_MS,
) -> dict:
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC)
    lang = language if language and language != "auto" else None
    chunks, duration = split_audio(filename, data, chunk_ms)

    label = speaker_numberer()
    paragraphs, cues, words = [], [], []
    detected_language = None

    for i, (chunk, offset) in enumerate(chunks):
        if progress:
            progress(i, len(chunks))
        file = (f"part{i + 1}.mp3", chunk)

        if model_id == "gpt-transcribe":
            kwargs = {"model": model_id, "file": file}
            if lang:
                kwargs["languages"] = [lang]
            if keyterms:
                kwargs["keywords"] = list(keyterms)
            r = _as_dict(client.audio.transcriptions.create(**kwargs))
            langs = r.get("languages") or []
            detected_language = detected_language or (langs[0].get("code") if langs else None)
            _append_text(paragraphs, r.get("text", ""))

        elif model_id == "gpt-4o-mini-transcribe":
            kwargs = {"model": model_id, "file": file, "response_format": "json"}
            if lang:
                kwargs["language"] = lang
            if keyterms:
                kwargs["prompt"] = "Vocabulary: " + ", ".join(keyterms)
            r = _as_dict(client.audio.transcriptions.create(**kwargs))
            _append_text(paragraphs, r.get("text", ""))

        elif model_id == "whisper-1":
            kwargs = {
                "model": model_id, "file": file, "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"], "temperature": 0,
            }
            if lang:
                kwargs["language"] = lang
            if keyterms:
                kwargs["prompt"] = "Vocabulary: " + ", ".join(keyterms)
            r = _as_dict(client.audio.transcriptions.create(**kwargs))
            detected_language = detected_language or r.get("language")
            for s in r.get("segments") or []:
                text = (s.get("text") or "").strip()
                if text:
                    cues.append({"speaker": None, "start": s["start"] + offset,
                                 "end": s["end"] + offset, "text": text})
            for w in r.get("words") or []:
                words.append({"word": w["word"].strip(), "sep": " ", "start": w["start"] + offset,
                              "end": w["end"] + offset, "speaker": None})

        elif model_id == "gpt-4o-transcribe-diarize":
            kwargs = {"model": model_id, "file": file, "response_format": "diarized_json",
                      "chunking_strategy": "auto"}
            if lang:
                kwargs["language"] = lang
            r = _as_dict(client.audio.transcriptions.create(**kwargs))
            for s in r.get("segments") or []:
                text = (s.get("text") or "").strip()
                if text:
                    cues.append({"speaker": label(s.get("speaker")),
                                 "start": s["start"] + offset, "end": s["end"] + offset, "text": text})

        else:
            raise ValueError(f"Unknown OpenAI model: {model_id}")

    if model_id == "gpt-4o-transcribe-diarize":
        paragraphs = merge_same_speaker(cues)
    elif model_id == "whisper-1":
        paragraphs = paragraphs_from_cues(cues)

    speakers = []
    for c in cues:
        if c["speaker"] and c["speaker"] not in speakers:
            speakers.append(c["speaker"])

    warning = None
    if model_id == "gpt-4o-transcribe-diarize" and len(chunks) > 1:
        warning = ("This recording was sent in 10-minute pieces; this model labels speakers per piece, "
                   "so 'Speaker 1' may not be the same person throughout. Scribe v2 keeps speakers consistent.")

    return {
        "language": detected_language or (lang or "unknown"),
        "duration": duration,
        "paragraphs": paragraphs,
        "cues": cues or None,
        "words": words or None,
        "speakers": speakers,
        "warning": warning,
    }


def _append_text(paragraphs: list, text: str):
    """Add one untimed piece's text to a single running paragraph."""
    text = (text or "").strip()
    if not text:
        return
    if paragraphs:
        paragraphs[-1]["text"] = smart_join(paragraphs[-1]["text"], text)
    else:
        paragraphs.append({"speaker": None, "start": None, "end": None, "text": text})
