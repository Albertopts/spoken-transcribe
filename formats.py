"""
Shared result shape and output builders (plain text, timed text, SRT, word JSON).

Every engine returns:
    {
      "language": str, "duration": float,
      "paragraphs": [{"speaker": str|None, "start": float|None, "end": float|None, "text": str}],
      "cues": [{"speaker", "start", "end", "text"}] | None,     # subtitle-sized, None = no timing
      "words": [{"word", "sep", "start", "end", "speaker"}] | None,
      "speakers": [str], "warning": str|None,
    }
"""

import json
import re

CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")

MAX_CUE_SEC = 7.0
MAX_CUE_WIDTH = 80   # CJK characters count double
SENTENCE_END = ("。", "！", "？", ".", "!", "?")


def smart_join(a: str, b: str) -> str:
    """Join two text pieces: no space between CJK characters, one space otherwise."""
    a, b = a.rstrip(), b.lstrip()
    if not a or not b:
        return a + b
    if CJK.match(a[-1]) or CJK.match(b[0]):
        return a + b
    return f"{a} {b}"


def _width(text: str) -> int:
    return sum(2 if CJK.match(ch) else 1 for ch in text)


def speaker_numberer():
    """Map raw speaker ids to "Speaker 1", "Speaker 2", ... in order of first appearance."""
    numbers = {}

    def label(raw):
        if raw is None:
            return None
        if raw not in numbers:
            numbers[raw] = len(numbers) + 1
        return f"Speaker {numbers[raw]}"

    return label


def paragraphs_from_words(words: list[dict]) -> list[dict]:
    """Group consecutive words by speaker into paragraphs, keeping original spacing."""
    paragraphs = []
    for w in words:
        if not paragraphs or paragraphs[-1]["speaker"] != w["speaker"]:
            paragraphs.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": ""})
        p = paragraphs[-1]
        p["text"] += w["word"] + w["sep"]
        p["end"] = w["end"]
    for p in paragraphs:
        p["text"] = p["text"].strip()
    return [p for p in paragraphs if p["text"]]


def cues_from_words(words: list[dict]) -> list[dict]:
    """Split words into subtitle cues: break on speaker change, length, or sentence end."""
    cues = []
    current = None
    for w in words:
        if current is not None:
            too_long = (w["end"] - current["start"] > MAX_CUE_SEC
                        or _width(current["text"] + w["word"]) > MAX_CUE_WIDTH)
            sentence_done = current["text"].rstrip().endswith(SENTENCE_END) and current["end"] - current["start"] > 1.5
            if w["speaker"] != current["speaker"] or too_long or sentence_done:
                cues.append(current)
                current = None
        if current is None:
            current = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": ""}
        current["text"] += w["word"] + w["sep"]
        current["end"] = w["end"]
    if current is not None:
        cues.append(current)
    for c in cues:
        c["text"] = c["text"].strip()
    return [c for c in cues if c["text"]]


def merge_same_speaker(segments: list[dict]) -> list[dict]:
    """Merge consecutive timed segments from the same speaker into paragraphs."""
    paragraphs = []
    for s in segments:
        if paragraphs and paragraphs[-1]["speaker"] == s["speaker"]:
            p = paragraphs[-1]
            p["text"] = smart_join(p["text"], s["text"])
            p["end"] = s["end"]
        else:
            paragraphs.append(dict(s))
    return paragraphs


def paragraphs_from_cues(cues: list[dict], max_sec: float = 60.0, gap_sec: float = 1.5) -> list[dict]:
    """Build paragraphs from untimed-speaker cues: break on a pause or after ~1 minute."""
    paragraphs = []
    for c in cues:
        if paragraphs:
            p = paragraphs[-1]
            if (c["speaker"] == p["speaker"] and c["start"] - p["end"] < gap_sec
                    and c["end"] - p["start"] < max_sec):
                p["text"] = smart_join(p["text"], c["text"])
                p["end"] = c["end"]
                continue
        paragraphs.append(dict(c))
    return paragraphs


def clock(seconds: float, millis: bool = False, sep: str = ",") -> str:
    ms_total = int(round(max(seconds, 0) * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}{sep}{ms:03d}" if millis else base


def to_text(result: dict, speaker_labels: bool, paragraph_times: bool) -> str:
    lines = []
    for p in result["paragraphs"]:
        prefix = ""
        if paragraph_times and p["start"] is not None:
            prefix += f"[{clock(p['start'])}] "
        if speaker_labels and p["speaker"]:
            prefix += f"{p['speaker']}: "
        lines.append(prefix + p["text"])
    return "\n\n".join(lines)


def to_srt(result: dict, speaker_labels: bool) -> str:
    blocks = []
    for i, c in enumerate(result["cues"] or [], start=1):
        text = f"{c['speaker']}: {c['text']}" if speaker_labels and c["speaker"] else c["text"]
        blocks.append(f"{i}\n{clock(c['start'], True)} --> {clock(c['end'], True)}\n{text}\n")
    return "\n".join(blocks)


def to_words_json(result: dict, speaker_labels: bool) -> str:
    """Word-level JSON in the same shape as the main app's transcript JSON."""
    words = []
    for w in result["words"] or []:
        item = {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        if speaker_labels and w["speaker"]:
            item["speaker"] = w["speaker"]
        words.append(item)
    payload = {
        "text": to_text(result, speaker_labels=False, paragraph_times=False),
        "words": words,
        "language": result["language"],
        "duration": result["duration"],
    }
    if speaker_labels and result["speakers"]:
        payload["speakers"] = result["speakers"]
    return json.dumps(payload, ensure_ascii=False, indent=2)
