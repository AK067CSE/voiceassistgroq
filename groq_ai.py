"""
groq_ai.py
──────────
Groq LLM — generates conversational responses spoken aloud via TTS.
Keeps a running message history so the conversation has memory.
"""
import os

try:
    import streamlit as st
    _api_key = st.secrets.get("GROQ_API_KEY")
except ImportError:
    _api_key = None

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=_api_key or os.getenv("GROQ_API_KEY", ""))

# In-memory conversation history  { session_id: [messages] }
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


def generate_response(prompt: str, session_id: str = "default") -> str:
    """
    Send prompt to Groq (llama-3.1-8b-instant) and return the text reply.
    Maintains conversation history per session_id.
    """
    if session_id not in _history:
        _history[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    _history[session_id].append({"role": "user", "content": prompt})

    completion = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=_history[session_id],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += (chunk.choices[0].delta.content or "")

    _history[session_id].append({"role": "assistant", "content": response})
    return response.strip()


def clear_history(session_id: str = "default"):
    """Wipe conversation memory for a session."""
    _history.pop(session_id, None)
