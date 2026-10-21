"""
Cloud transcription page (Streamlit Community Cloud).
Upload audio → pick a model (ElevenLabs Scribe v2 by default) → download text.

Deploy: share.streamlit.io → Create app → this repo, branch main,
main file path app.py. Secrets (Advanced settings):
    ELEVENLABS_API_KEY = "..."
    APP_PASSWORD = "..."
    OPENAI_API_KEY = "..."      # optional, only for the OpenAI models
"""

import hmac
import re
from pathlib import Path

import streamlit as st

import catalog as C
import openai_stt
import scribe
from errors import classify_error
from formats import to_srt, to_text, to_words_json

AUDIO_TYPES = ["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "opus", "webm", "mov", "mpeg", "mpga"]
MAX_KEYTERMS = 1000
MAX_KEYTERM_LEN = 49

HELP = {
    C.ELEVENLABS: {
        "credits": (
            "⛔ **ElevenLabs credit limit reached — this file was not transcribed.**\n\n"
            "What to do (ElevenLabs' message below says which one applies):\n"
            "1. **This app's key hit its limit** → elevenlabs.io → Developers → API Keys → "
            "edit the key used by this app → raise *Per credit refresh period* "
            "(or wait until the limit refreshes).\n"
            "2. **Your account is out of credits** → elevenlabs.io → Subscription → "
            "top up / turn on usage-based billing / upgrade the plan.\n\n"
            "Then tap **Transcribe** again — your upload is kept."
        ),
        "permissions": (
            "⛔ **This app's ElevenLabs key isn't allowed to use Speech to Text.**\n\n"
            "elevenlabs.io → Developers → API Keys → edit the key → set *Speech to Text* to **Access**, "
            "then tap **Transcribe** again."
        ),
        "key": (
            "⛔ **ElevenLabs API key rejected** (deleted, expired or mistyped).\n\n"
            "Create a new key in elevenlabs.io → Developers → API Keys (Speech to Text only), then paste it "
            "into this app's Streamlit **Settings → Secrets** as `ELEVENLABS_API_KEY`."
        ),
    },
    C.OPENAI: {
        "credits": (
            "⛔ **OpenAI account out of credit — this file was not transcribed.**\n\n"
            "platform.openai.com → Settings → Billing → add credit (or raise the project's monthly budget), "
            "then tap **Transcribe** again — your upload is kept."
        ),
        "permissions": (
            "⛔ **This OpenAI key can't use this model.**\n\n"
            "platform.openai.com → API keys → edit the key → allow **Audio** (`/v1/audio`) — or check the "
            "project has access to this model. Then tap **Transcribe** again."
        ),
        "key": (
            "⛔ **OpenAI API key rejected** (deleted, revoked or mistyped).\n\n"
            "Create a new key at platform.openai.com → API keys, then paste it into this app's Streamlit "
            "**Settings → Secrets** as `OPENAI_API_KEY`."
        ),
    },
}
COMMON_HELP = {
    "busy": "⏳ **The service is busy or the connection dropped.** Wait a minute, then tap **Transcribe** again.",
    "other": "⛔ **Transcription failed.**",
}

st.set_page_config(page_title="Transcribe", page_icon="🎙️", layout="centered")


def get_secret(name: str):
    """Read a secret; None when it, or the whole secrets file, is missing."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def require_password():
    """Stop the page until the correct password is entered."""
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD is not set in this app's secrets.")
        st.stop()
    if st.session_state.get("authed"):
        return

    st.title("🎙️ Transcribe")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", use_container_width=True)
    if submitted:
        if hmac.compare_digest(password.encode(), str(expected).encode()):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password")
    st.stop()


def show_error(err: Exception, provider: str):
    """Explain a failure in plain words, with what to do next."""
    kind, message = classify_error(err)
    st.error(HELP[provider].get(kind) or COMMON_HELP[kind])
    if message:
        st.caption(f"{C.PROVIDER_NAMES[provider]} says: {message}")


def parse_keyterms(raw: str) -> tuple:
    terms = []
    for t in re.split(r"[\n,，、]", raw or ""):
        t = t.strip()[:MAX_KEYTERM_LEN]
        if t and t not in terms:
            terms.append(t)
    return tuple(terms[:MAX_KEYTERMS])


def comparison_table() -> str:
    rows = ["| Model | $/hour | Speakers | Timestamps | Why pick it |", "|---|---|---|---|---|"]
    for m in C.MODELS:
        ts = "word" if m["word_timestamps"] else ("sentence" if m["timed"] else "—")
        name = f"{C.PROVIDER_NAMES[m['provider']]} {m['name']}"
        if m["deprecated"]:
            name += " *(old)*"
        rows.append(f"| {name} | ${m['usd_per_hour']:.2f} | {'✓' if m['speakers'] else '—'} | {ts} | {m['why']} |")
    return "\n".join(rows)


@st.cache_data(show_spinner=False, max_entries=10, ttl=24 * 3600)
def cached_transcribe(model_id: str, filename: str, data: bytes, language: str,
                      num_speakers: int, keyterms: tuple, no_verbatim: bool, tag_events: bool) -> dict:
    # Cached by file content + recognition options: re-uploading the same file after the
    # phone drops the connection returns the result without paying again.
    m = C.MODELS_BY_ID[model_id]
    api_key = get_secret(C.SECRET_NAMES[m["provider"]])
    if m["provider"] == C.ELEVENLABS:
        return scribe.transcribe(
            api_key, filename, data, model_id=model_id, language=language,
            num_speakers=num_speakers or None, keyterms=keyterms,
            no_verbatim=no_verbatim, tag_audio_events=tag_events,
        )
    return openai_stt.transcribe(api_key, filename, data, model_id, language=language, keyterms=keyterms)


require_password()

st.title("🎙️ Transcribe")
st.caption("Upload audio → text with speaker labels. Defaults: ElevenLabs Scribe v2, no timestamps.")

uploaded = st.file_uploader("Audio file", type=AUDIO_TYPES)

with st.expander("⚙️ Settings — model, language, output"):
    st.markdown("**Model**")
    model_ids = [m["id"] for m in C.MODELS]
    model_id = st.selectbox(
        "Model", options=model_ids, index=model_ids.index(C.DEFAULT_MODEL_ID),
        format_func=C.model_label, label_visibility="collapsed",
    )
    m = C.MODELS_BY_ID[model_id]
    provider = m["provider"]
    st.caption(m["why"] + (f" *Deprecated by OpenAI — {C.OPENAI_RETIREMENT}.*"
                           if m["deprecated"] and provider == C.OPENAI else ""))
    if st.toggle("Compare all models"):
        st.markdown(comparison_table())
    if not get_secret(C.SECRET_NAMES[provider]):
        st.warning(f"Add `{C.SECRET_NAMES[provider]}` to this app's Secrets to use {C.PROVIDER_NAMES[provider]} models.")

    st.markdown("**Recognition**")
    language = st.selectbox("Language", options=list(C.LANGUAGES), format_func=C.LANGUAGES.get,
                            help="Keep auto-detect when speakers mix languages — forcing one hurts the others.")
    num_speakers = 0
    if provider == C.ELEVENLABS:
        num_speakers = st.number_input("Number of speakers (0 = auto-detect)", 0, 32, 0, 1)
    terms_help = {
        "scribe_v2": "Biases recognition toward these words. ElevenLabs add-on: +$0.05 per audio hour.",
        "scribe_v1": "Not supported by Scribe v1 — ignored.",
        "gpt-4o-transcribe-diarize": "Not supported by this model — ignored.",
    }.get(model_id, "Passed to the model as vocabulary hints. No extra cost.")
    keyterms_raw = st.text_area("Names & special terms (optional)", placeholder="One per line, e.g.\nCostco\nKirkland Signature",
                                help=terms_help, height=80)
    no_verbatim = tag_events = False
    if provider == C.ELEVENLABS:
        no_verbatim = st.checkbox("Remove filler words (um, uh, false starts)", value=False,
                                  disabled=model_id != "scribe_v2", help="Scribe v2 only.")
        tag_events = st.checkbox("Mark sounds like (laughter), (applause)", value=False)

    st.markdown("**Output**")
    speaker_labels = st.checkbox("Speaker labels (Speaker 1, Speaker 2…)", value=True, disabled=not m["speakers"],
                                 help=None if m["speakers"] else "This model can't tell speakers apart.")
    paragraph_times = st.checkbox("Timestamp at the start of each paragraph", value=False, disabled=not m["timed"],
                                  help=None if m["timed"] else "This model returns no timestamps.")
    want_srt = st.checkbox("Subtitles file (.srt)", value=False, disabled=not m["timed"],
                           help=None if m["timed"] else "This model returns no timestamps.")
    want_words = st.checkbox("Timestamp for every word (.json)", value=False, disabled=not m["word_timestamps"],
                             help="Same JSON shape as the remix app's transcripts." if m["word_timestamps"]
                             else "This model has no word-level timestamps.")

keyterms = parse_keyterms(keyterms_raw)
if model_id in ("scribe_v1", "gpt-4o-transcribe-diarize"):
    keyterms = ()
key_ok = bool(get_secret(C.SECRET_NAMES[provider]))

st.caption(f"Model: **{C.model_label(model_id)}**")
if st.button("🎙️ Transcribe", type="primary", use_container_width=True, disabled=not (uploaded and key_ok)):
    try:
        with st.spinner("Transcribing… keep this page open. Long recordings take a few minutes."):
            result = cached_transcribe(model_id, uploaded.name, uploaded.getvalue(), language,
                                       int(num_speakers), keyterms, no_verbatim, tag_events)
        st.session_state.result = result
        st.session_state.result_meta = {
            "file_id": uploaded.file_id, "filename": uploaded.name,
            "model_id": model_id, "keyterms": bool(keyterms),
        }
    except Exception as e:
        show_error(e, provider)
        st.stop()

result = st.session_state.get("result")
meta = st.session_state.get("result_meta")
if result and meta and uploaded and meta["file_id"] == uploaded.file_id:
    rm = C.MODELS_BY_ID[meta["model_id"]]
    if meta["model_id"] != model_id:
        st.info(f"Showing the result from **{C.model_label(meta['model_id'])}**. Tap Transcribe to run the selected model.")

    minutes = result["duration"] / 60
    cost = C.estimate_cost(meta["model_id"], result["duration"], meta["keyterms"])
    speakers_txt = f"{len(result['speakers'])} speaker(s), " if rm["speakers"] else ""
    st.success(f"✅ Done — {minutes:.1f} min, {speakers_txt}language: {result['language']}, est. cost ${cost:.3f}")
    if result.get("warning"):
        st.warning(result["warning"])

    use_speakers = speaker_labels and rm["speakers"]
    text = to_text(result, use_speakers, paragraph_times and rm["timed"])
    stem = Path(meta["filename"]).stem

    st.download_button("⬇️ Download transcript (.txt)", data=(text + "\n").encode("utf-8"),
                       file_name=f"{stem}_transcript.txt", mime="text/plain",
                       type="primary", use_container_width=True)
    if want_srt and rm["timed"] and result["cues"]:
        st.download_button("⬇️ Download subtitles (.srt)", data=to_srt(result, use_speakers).encode("utf-8"),
                           file_name=f"{stem}.srt", mime="text/plain", use_container_width=True)
    if want_words and rm["word_timestamps"] and result["words"]:
        st.download_button("⬇️ Download word timestamps (.json)",
                           data=to_words_json(result, use_speakers).encode("utf-8"),
                           file_name=f"{stem}_transcript_words.json", mime="application/json",
                           use_container_width=True)

    st.code(text or "(no speech detected)", language=None, wrap_lines=True)
