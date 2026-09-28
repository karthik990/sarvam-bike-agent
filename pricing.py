"""Sarvam list prices (INR) from https://docs.sarvam.ai/api/getting-started/pricing (checked Sep 2026).
Override via env if your plan differs. LLM prices are per 1M tokens."""
import os

LLM = {  # model: (input, output) ₹ per 1M tokens
    "sarvam-105b": (29.28, 73.2),
    "sarvam-105b-conversations": (29.28, 73.2),
    "gemma4": (36.6, 91.5),
}
STT_PER_HOUR = float(os.getenv("PRICE_STT_PER_HOUR", "30"))      # Saaras transcribe / translate
TTS_PER_10K_CHARS = float(os.getenv("PRICE_TTS_PER_10K", "30"))  # Bulbul v3
FREE_CREDITS = float(os.getenv("SARVAM_BUDGET_INR", "100"))      # new accounts get ₹100


def llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = LLM.get(model, (29.28, 73.2))
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


def stt_cost(seconds: float) -> float:
    return seconds / 3600 * STT_PER_HOUR


def tts_cost(chars: int) -> float:
    return chars / 10_000 * TTS_PER_10K_CHARS


def inr(x: float) -> str:
    return f"₹{x:.4f}" if x < 1 else f"₹{x:.2f}"
