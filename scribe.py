"""
ElevenLabs Scribe v2 transcription → plain text with speaker turns, no timestamps.
Speakers are labelled "Speaker 1", "Speaker 2", ... in order of first appearance.
"""

from typing import Optional

from elevenlabs import ElevenLabs

# Long recordings can take several minutes; the SDK default is 240s.
REQUEST_TIMEOUT_SEC = 1800


def format_speaker_turns(words) -> tuple[str, list[str]]:
    """
    Group Scribe tokens into "Speaker N: ..." paragraphs.

    Speakers are numbered in order of first appearance. Spacing tokens are
    kept as-is so English keeps its spaces and Chinese stays unspaced.
    """
    speaker_numbers = {}
    turns = []  # [label, [token texts]]
    for w in (words or []):
        if w.type == "audio_event":
            continue
        if w.type == "word":
            raw_speaker = w.speaker_id or "speaker_0"
            if raw_speaker not in speaker_numbers:
                speaker_numbers[raw_speaker] = len(speaker_numbers) + 1
            label = f"Speaker {speaker_numbers[raw_speaker]}"
            if not turns or turns[-1][0] != label:
                turns.append([label, []])
        if turns:
            turns[-1][1].append(w.text)

    paragraphs = []
    for label, parts in turns:
        turn_text = "".join(parts).strip()
        if turn_text:
            paragraphs.append(f"{label}: {turn_text}")

    speakers = [f"Speaker {n}" for n in sorted(speaker_numbers.values())]
    return "\n\n".join(paragraphs), speakers


def transcribe_to_text(
    api_key: str,
    filename: str,
    data: bytes,
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
) -> dict:
    """
    Transcribe audio bytes with ElevenLabs Scribe v2 (diarization + mixed-language).

    Returns:
        Dictionary with text, speakers, language, duration (seconds, from last word)
    """
    client = ElevenLabs(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC)

    kwargs = {
        "model_id": "scribe_v2",
        "diarize": True,
        "tag_audio_events": False,
        "timestamps_granularity": "word",  # speaker_id is attached per word
        "file": (filename, data),
    }
    if language and language != "auto":
        kwargs["language_code"] = language
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    response = client.speech_to_text.convert(**kwargs)

    text, speakers = format_speaker_turns(response.words)
    if not text:
        text = (response.text or "").strip()

    ends = [w.end for w in (response.words or []) if getattr(w, "end", None) is not None]

    return {
        "text": text,
        "speakers": speakers,
        "language": getattr(response, "language_code", None) or "unknown",
        "duration": max(ends) if ends else 0.0,
    }
