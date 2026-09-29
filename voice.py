"""Voice question -> text, kept separate from the UI so it can be tested without a microphone.

Uses Saaras in TRANSCRIBE mode with automatic language detection. (TRANSLATE mode only converts
Indic speech to English, so an English question could come back empty and silently do nothing.)
Non-English questions are translated for retrieval by the agent's planner step, which already runs
for non-English input, so there is no second speech-to-text call.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent import LANG_NAMES, detect_lang

MAX_SECONDS = 30  # Sarvam REST speech-to-text limit


@dataclass
class VoiceResult:
    question: str | None          # text to send to the agent (None = nothing usable)
    english_query: str | None     # set only when the speech was English
    lang: str | None              # BCP-47 code of the spoken language, e.g. hi-IN
    message: str = ""             # user-facing status / error
    level: str = "info"           # info | warning | error


def audio_seconds(audio: bytes) -> float | None:
    try:
        import io
        import wave
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def voice_to_question(client, audio: bytes) -> VoiceResult:
    if client is None:
        return VoiceResult(None, None, None, "Voice questions need the Sarvam API. Turn off Offline mode.", "warning")
    if not audio or len(audio) < 1000:
        return VoiceResult(None, None, None, "Didn't catch any audio. Please record again.", "warning")
    secs = audio_seconds(audio)
    if secs is not None and secs > MAX_SECONDS:
        return VoiceResult(None, None, None, f"That clip is {secs:.0f} s; please keep voice questions under "
                                             f"{MAX_SECONDS} seconds.", "warning")
    try:
        text, code = client.transcribe(audio)
    except Exception as e:  # show the reason instead of failing silently
        msg = str(e)
        hint = " (clip too long or unsupported format?)" if "422" in msg else ""
        return VoiceResult(None, None, None, f"Speech-to-text failed{hint}: {msg[:160]}", "error")
    text = (text or "").strip()
    if not text:
        return VoiceResult(None, None, None, "Couldn't make out any words. Please speak a bit closer to the mic "
                                             "and try again.", "warning")
    lang = code if code in LANG_NAMES else detect_lang(text)
    english = lang == "en-IN" or detect_lang(text) == "en-IN"
    return VoiceResult(text, text if english else None, "en-IN" if english else lang, f"🎙️ Heard: {text}")
