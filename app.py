"""
app.py  —  Voice Chat  (Streamlit + Deepgram STT/TTS + Groq)
─────────────────────────────────────────────────────────────
Flow per turn:
  1. User records audio  →  Deepgram STT (nova-3)         →  transcript
  2. Transcript          →  Groq llama-3.1-8b-instant     →  reply text
  3. Reply text          →  Deepgram TTS (aura-2)         →  wav audio
  4. Play audio + show chat bubbles

Run:
  streamlit run app.py
"""

import os
import uuid
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions  # noqa: F401
from groq_ai import generate_response, clear_history

# ── Resolve correct SpeakOptions class across SDK versions ───────────────────
try:
    from deepgram import SpeakOptions as _SpeakOpts  # SDK v3.2–3.x
except ImportError:
    try:
        from deepgram import SpeakRESTOptions as _SpeakOpts  # SDK v3.3+
    except ImportError:
        _SpeakOpts = None  # fallback: pass dict directly

# ── Page config (MUST be first Streamlit command) ─────────────────────────────
st.set_page_config(
    page_title="Talking Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── API Key Loading (AFTER st.set_page_config) ────────────────────────────────
def load_api_keys():
    """Load API keys from Streamlit secrets or environment variables."""
    try:
        dg_key   = st.secrets.get("DEEPGRAM_API_KEY", "")
        groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        dg_key   = os.getenv("DEEPGRAM_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
    return str(dg_key).strip(), str(groq_key).strip()

DG_API_KEY, GROQ_API_KEY = load_api_keys()

if not DG_API_KEY:
    st.error("❌ Missing `DEEPGRAM_API_KEY` in secrets.toml or Streamlit Cloud Secrets.")
    st.stop()
if not GROQ_API_KEY:
    st.error("❌ Missing `GROQ_API_KEY` in secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

# ── Deepgram voice options ────────────────────────────────────────────────────
VOICES = {
    "Asteria — Warm Female":   "aura-2-asteria-en",
    "Luna — Soft Female":      "aura-2-luna-en",
    "Stella — Upbeat Female":  "aura-2-stella-en",
    "Athena — British Female": "aura-2-athena-en",
    "Hera — Confident Female": "aura-2-hera-en",
    "Orion — Deep Male":       "aura-2-orion-en",
    "Arcas — Casual Male":     "aura-2-arcas-en",
    "Perseus — Neutral Male":  "aura-2-perseus-en",
    "Angus — Irish Male":      "aura-2-angus-en",
    "Orpheus — Rich Male":     "aura-2-orpheus-en",
    "Helios — Energetic Male": "aura-2-helios-en",
    "Zeus — Bold Male":        "aura-2-zeus-en",
}

# ── Deepgram helpers ──────────────────────────────────────────────────────────

def stt(audio_bytes: bytes) -> str:
    """audio bytes → transcript via Deepgram nova-3 (SDK v3 compatible)"""
    try:
        deepgram = DeepgramClient(api_key=DG_API_KEY)
        payload = {"buffer": audio_bytes}
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            punctuate=True,
        )

        # Try newer 'listen.rest' path first (SDK v3.3+), fall back to 'listen.prerecorded'
        try:
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        except AttributeError:
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        transcript = response.results.channels[0].alternatives[0].transcript.strip()
        return transcript

    except Exception as e:
        st.error(f"🔇 STT Error: {type(e).__name__} — {e}")
        import traceback
        st.code(traceback.format_exc())
        return ""


def tts(text: str, voice_model: str) -> str:
    """text → wav file via Deepgram aura-2 (SDK v3 compatible), returns unique filename"""
    output_wav = f"output_{uuid.uuid4().hex[:8]}.wav"
    try:
        deepgram = DeepgramClient(api_key=DG_API_KEY)

        speak_payload = {"text": text}
        speak_options = {"model": voice_model, "encoding": "linear16", "container": "wav"}

        # Build options object if available, else pass raw dict
        if _SpeakOpts is not None:
            opts_obj = _SpeakOpts(**speak_options)
        else:
            opts_obj = speak_options

        response = deepgram.speak.v("1").save(output_wav, speak_payload, opts_obj)

        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 0:
            return output_wav
        else:
            st.error("🔊 Audio file was not created or is empty.")
            return ""

    except Exception as e:
        st.error(f"🔊 TTS Error: {type(e).__name__} — {e}")
        import traceback
        st.code(traceback.format_exc())
        return ""


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recorder_turn" not in st.session_state:
    st.session_state.recorder_turn = 0
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_audio_size" not in st.session_state:
    st.session_state.last_audio_size = 0

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Fraunces:ital,wght@0,900;1,700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080b10;
    color: #d8dff0;
}

.main .block-container {
    max-width: 720px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

.va-header {
    text-align: center;
    padding: 1.8rem 0 1rem;
}
.va-title {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: #f0f4ff;
    line-height: 1.1;
}
.va-title span {
    background: linear-gradient(120deg, #f97316, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.va-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #3a4570;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.va-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2540, transparent);
    margin: 1.2rem 0 1.6rem;
}

.brow { display: flex; margin-bottom: 0.9rem; align-items: flex-end; gap: 10px; }
.brow.user  { flex-direction: row-reverse; }
.brow.agent { flex-direction: row; }

.av {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.av.user  { background: #1a2840; }
.av.agent { background: #200e1a; }

.bub {
    max-width: 72%; border-radius: 16px;
    padding: 11px 15px; font-size: 0.88rem; line-height: 1.65;
    word-break: break-word;
}
.bub.user  {
    background: #152236;
    border-bottom-right-radius: 4px;
    color: #93c5fd;
}
.bub.agent {
    background: #1a0e1a;
    border-bottom-left-radius: 4px;
    color: #f9a8d4;
    border: 1px solid #2d1130;
}

.empty-hint {
    text-align: center; color: #252d48;
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
    margin-top: 3.5rem; letter-spacing: 0.5px;
}

.rec-anchor {
    position: fixed; bottom: 0; left: 0; right: 0;
    padding: 1rem 0 1.2rem;
    background: linear-gradient(to top, #080b10 60%, transparent);
    display: flex; flex-direction: column;
    align-items: center; gap: 6px; z-index: 99;
}
.rec-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #252d48;
    letter-spacing: 1.5px; text-transform: uppercase;
}

.groq-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(249,115,22,.1); color: #fb923c;
    border: 1px solid rgba(249,115,22,.2);
    border-radius: 20px; padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
}
.dg-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(236,72,153,.1); color: #f472b6;
    border: 1px solid rgba(236,72,153,.2);
    border-radius: 20px; padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
}

audio {
    width: 100%; height: 34px; margin-top: 8px; border-radius: 8px;
    filter: invert(0.9) sepia(0.4) hue-rotate(290deg);
}

section[data-testid="stSidebar"] {
    background: #0d0f18;
    border-right: 1px solid #14192b;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }

.stSpinner > div { border-top-color: #f97316 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    voice_name = st.selectbox(
        "Deepgram Voice",
        options=list(VOICES.keys()),
        index=0,
    )
    voice_model = VOICES[voice_name]

    st.markdown("---")
    st.markdown(
        f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;
                    color:#3a4570;line-height:2.2;'>
        STT &nbsp;&nbsp;nova-3<br>
        LLM &nbsp;&nbsp;llama-3.1-8b-instant<br>
        TTS &nbsp;&nbsp;aura-2<br>
        Voice &nbsp;{voice_model}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        clear_history(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.recorder_turn = 0
        st.session_state.last_audio_size = 0
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="va-header">
  <div class="va-title">Talking Assistant <span>🎙️</span></div>
  <div class="va-sub">Groq · Deepgram · nova-3 · aura-2</div>
</div>
<div class="va-line"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<div class="groq-badge">⚡ Groq llama-3.1-8b-instant</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="dg-badge">🎤 Deepgram nova-3</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="dg-badge">🔊 Deepgram aura-2</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown(
        '<p class="empty-hint">Press the mic button below to start talking</p>',
        unsafe_allow_html=True,
    )

for turn in st.session_state.chat_history:
    role   = turn["role"]
    text   = turn["text"]
    audio  = turn.get("audio_file")
    avatar = "🧑" if role == "user" else "🤖"

    st.markdown(f"""
    <div class="brow {role}">
      <div class="av {role}">{avatar}</div>
      <div class="bub {role}">{text}</div>
    </div>
    """, unsafe_allow_html=True)

    if role == "agent" and audio and os.path.exists(audio):
        with open(audio, "rb") as f:
            st.audio(f.read(), format="audio/wav")

# ── Recorder (pinned bottom) ──────────────────────────────────────────────────
recorder_key = f"recorder_{st.session_state.recorder_turn}"

st.markdown('<div class="rec-anchor">', unsafe_allow_html=True)

audio_bytes = audio_recorder(
    text="",
    recording_color="#f97316",
    neutral_color="#1a2540",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.5,
    sample_rate=16000,
    key=recorder_key,
)

hint = "processing… please wait" if st.session_state.processing else "click to record · silence stops automatically"
st.markdown(f'<div class="rec-hint">{hint}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Process new recording ─────────────────────────────────────────────────────
# Guard: only process if audio is new (different size) and large enough to be real speech
MIN_AUDIO_BYTES = 1000  # ignore tiny/empty buffers

if (
    audio_bytes
    and not st.session_state.processing
    and len(audio_bytes) > MIN_AUDIO_BYTES
    and len(audio_bytes) != st.session_state.last_audio_size
):
    st.session_state.last_audio_size = len(audio_bytes)
    st.session_state.processing = True

    # Step 1 — STT
    with st.spinner("Transcribing…"):
        transcript = stt(audio_bytes)

    if not transcript:
        st.warning("Couldn't catch that — please try again.")
        st.session_state.processing = False
        st.session_state.recorder_turn += 1
        st.rerun()

    st.session_state.chat_history.append({"role": "user", "text": transcript})

    # Step 2 — Groq LLM
    with st.spinner("Thinking…"):
        reply = generate_response(transcript, st.session_state.session_id, GROQ_API_KEY)

    if not reply:
        reply = "Sorry, I had trouble with that. Please try again."

    # Step 3 — TTS
    with st.spinner("Speaking…"):
        wav_file = tts(reply, voice_model)

    st.session_state.chat_history.append({
        "role":       "agent",
        "text":       reply,
        "audio_file": wav_file,
    })

    st.session_state.recorder_turn += 1
    st.session_state.processing = False
    st.rerun()
