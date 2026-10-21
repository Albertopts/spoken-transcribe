"""
ElevenLabs Scribe transcription → shared result shape (see formats.py).
"""

from typing import Optional

from elevenlabs import ElevenLabs

from formats import cues_from_words, paragraphs_from_words, speaker_numberer

# Long recordings can take several minutes; the SDK default is 240s.
REQUEST_TIMEOUT_SEC = 1800


def transcribe(
    api_key: str,
    filename: str,
    data: bytes,
    model_id: str = "scribe_v2",
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    keyterms: tuple = (),
    no_verbatim: bool = False,
    tag_audio_events: bool = False,
) -> dict:
    client = ElevenLabs(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC)

    kwargs = {
        "model_id": model_id,
        "diarize": True,
        "tag_audio_events": tag_audio_events,
        "timestamps_granularity": "word",  # speaker_id is attached per word
        "file": (filename, data),
    }
    if language and language != "auto":
        kwargs["language_code"] = language
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    if model_id == "scribe_v2":
        if keyterms:
            kwargs["keyterms"] = list(keyterms)
        if no_verbatim:
            kwargs["no_verbatim"] = True

    response = client.speech_to_text.convert(**kwargs)

    # Words carry their trailing spacing in "sep", so joining word + sep
    # reproduces Scribe's spacing (English spaced, Chinese unspaced).
    label = speaker_numberer()
    words = []
    for w in (response.words or []):
        if w.type == "spacing":
            if words:
                words[-1]["sep"] += w.text
            continue
        if w.type == "audio_event" and not tag_audio_events:
            continue
        if w.type == "audio_event" and words:
            speaker = words[-1]["speaker"]  # keep a tag inside the current speaker's turn
        else:
            speaker = label(w.speaker_id or "speaker_0")
        words.append({
            "word": w.text,
            "sep": "",
            "start": float(w.start or 0.0),
            "end": float(w.end or w.start or 0.0),
            "speaker": speaker,
        })

    paragraphs = paragraphs_from_words(words)
    if not paragraphs and (response.text or "").strip():
        paragraphs = [{"speaker": None, "start": None, "end": None, "text": response.text.strip()}]

    speakers = []
    for w in words:
        if w["speaker"] and w["speaker"] not in speakers:
            speakers.append(w["speaker"])

    return {
        "language": getattr(response, "language_code", None) or "unknown",
        "duration": max((w["end"] for w in words), default=0.0),
        "paragraphs": paragraphs,
        "cues": cues_from_words(words) if words else None,
        "words": words or None,
        "speakers": speakers,
        "warning": None,
    }
