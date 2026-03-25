"""
groq_ai.py
──────────
Groq LLM — generates conversational responses using llama-3.1-8b-instant.
"""
from groq import Groq

# ✅ No module-level secret access - API key passed as parameter
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


def get_groq_client(api_key: str) -> Groq:
    """Create Groq client with provided API key."""
    return Groq(api_key=api_key)


def generate_response(prompt: str, session_id: str, api_key: str) -> str:
    """
    Send prompt to Groq and return the text reply.
    API key is passed as parameter (not loaded at module level).
    """
    if session_id not in _history:
        _history[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    _history[session_id].append({"role": "user", "content": prompt})

    try:
        client = get_groq_client(api_key)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=_history[session_id],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )

        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                response += delta

        _history[session_id].append({"role": "assistant", "content": response})
        return response.strip()

    except Exception as e:
        print(f"❌ Groq Error: {type(e).__name__} — {e}")
        return ""


def clear_history(session_id: str = "default"):
    """Wipe conversation memory for a session."""
    _history.pop(session_id, None)
