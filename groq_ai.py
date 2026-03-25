"""
groq_ai.py
──────────
Groq LLM — generates conversational responses using llama-3.1-8b-instant.
"""
import os
from groq import Groq

# ── API Key Loading (Works with Streamlit Cloud secrets OR .env) ─────────────
try:
    import streamlit as st
    # Try Streamlit secrets first (for Streamlit Cloud)
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    # Fallback to environment variables (for local development)
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Strip whitespace and validate
GROQ_API_KEY = str(GROQ_API_KEY).strip()

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in secrets.toml or .env file")

_client = Groq(api_key=GROQ_API_KEY)
_history: dict[str, list] = {}

SYSTEM_PROMPT = """You are a friendly voice assistant. Your replies will be spoken aloud
so always write for the ear — short, natural, conversational.
Rules:
- Maximum 2 sentences per reply
- Use contractions and casual language
- No bullet points, no markdown, no lists
- Get straight to the key fact or answer
- Sound warm and human, not robotic
"""

MODEL_NAME = "llama-3.1-8b-instant"


def generate_response(prompt: str, session_id: str = "default") -> str:
    if session_id not in _history:
        _history[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    _history[session_id].append({"role": "user", "content": prompt})

    try:
        completion = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=_history[session_id],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        _history[session_id].append({"role": "assistant", "content": response})
        return response.strip()
    except Exception as e:
        print(f"❌ Groq Error: {type(e).__name__} — {e}")
        return ""


def clear_history(session_id: str = "default"):
    _history.pop(session_id, None)
