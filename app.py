"""
Cloud transcription page (Streamlit Community Cloud).
Upload audio → ElevenLabs Scribe v2 → download "Speaker N: ..." text. No timestamps.

Deploy: share.streamlit.io → Create app → this repo, branch main,
main file path app.py. Secrets (Advanced settings):
    ELEVENLABS_API_KEY = "..."
    APP_PASSWORD = "..."
"""

import hmac
from pathlib import Path

import streamlit as st

from scribe import transcribe_to_text

SCRIBE_USD_PER_HOUR = 0.22
AUDIO_TYPES = ["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "opus", "webm", "mov"]

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


@st.cache_data(show_spinner=False, max_entries=10, ttl=24 * 3600)
def cached_transcribe(filename: str, data: bytes, language: str, num_speakers: int) -> dict:
    # Cached by file content + options: re-uploading the same file after the
    # phone drops the connection returns the result without paying again.
    return transcribe_to_text(
        get_secret("ELEVENLABS_API_KEY"),
        filename,
        data,
        language=language,
        num_speakers=num_speakers or None,
    )


require_password()

if not get_secret("ELEVENLABS_API_KEY"):
    st.error("ELEVENLABS_API_KEY is not set in this app's secrets.")
    st.stop()

st.title("🎙️ Transcribe")
st.caption("Upload audio → text with speaker labels. ElevenLabs Scribe v2, ~$0.22 per audio hour.")

with st.expander("Options"):
    language = st.selectbox(
        "Language",
        options=["auto", "zh", "en", "de"],
        index=0,
        help="Keep 'auto' when speakers mix languages.",
    )
    num_speakers = st.number_input("Number of speakers (0 = auto-detect)", 0, 32, 0, 1)

uploaded = st.file_uploader("Audio file", type=AUDIO_TYPES)

if uploaded:
    try:
        with st.spinner("Transcribing… keep this page open. Long recordings take a few minutes."):
            result = cached_transcribe(uploaded.name, uploaded.getvalue(), language, int(num_speakers))
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

    minutes = result["duration"] / 60
    st.success(
        f"✅ Done — {minutes:.1f} min, {len(result['speakers'])} speaker(s), "
        f"language: {result['language']}, est. cost ${minutes / 60 * SCRIBE_USD_PER_HOUR:.3f}"
    )
    st.download_button(
        "⬇️ Download transcript (.txt)",
        data=(result["text"] + "\n").encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_transcript.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True,
    )
    st.code(result["text"] or "(no speech detected)", language=None, wrap_lines=True)
