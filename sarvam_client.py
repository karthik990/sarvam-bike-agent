"""Thin wrappers over Sarvam REST APIs (raw HTTP, so SDK version drift can't break the demo).

Token-frugal by design:
- reasoning is OFF by default (thinking tokens are billed as completion tokens)
- small max_tokens caps on every call
- images are downscaled to <=512px JPEG before upload
- every call's `usage` is accumulated so the UI can show exactly what was spent

Models used
- sarvam-105b      : grounded answers (the model behind Sarvam Indus)
- gemma4 (v2 beta) : image -> symptom description (only when a photo is attached)
- saaras:v3        : speech -> English text (mode=translate, so no extra LLM translation call)
- bulbul:v3        : text-to-speech (only on explicit click)
"""
from __future__ import annotations

import base64
import io
import os
import re
import time

import requests

import pricing

BASE = "https://api.sarvam.ai"
CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
VISION_MODEL = os.getenv("SARVAM_VISION_MODEL", "gemma4")


class SarvamError(RuntimeError):
    pass


def _drop_half_items(obj):
    """Recursively drop the last element of lists of cited items ({text, page}) when it lacks a page.
    Never touches container lists like 'parts' (their elements have no 'text' key)."""
    if isinstance(obj, dict):
        for v in obj.values():
            _drop_half_items(v)
    elif isinstance(obj, list):
        if obj and isinstance(obj[-1], dict) and "text" in obj[-1] and not obj[-1].get("page"):
            obj.pop()
        for v in obj:
            _drop_half_items(v)


def parse_json_loose(raw: str):
    """Strict json first; then json-repair (handles unescaped quotes, missing commas, truncation)."""
    import json
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        import json_repair
        return json_repair.loads(raw)
    except Exception:
        return {}


def shrink_image(data: bytes, max_side: int = 512) -> tuple[bytes, str]:
    """Downscale + JPEG-compress; falls back to original bytes if Pillow is missing."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/jpeg"


def _wav_seconds(b: bytes) -> float:
    try:
        import wave
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 30.0  # REST STT max; conservative estimate


class Sarvam:
    def __init__(self, api_key: str, timeout: int = 90):
        if not api_key:
            raise SarvamError("Missing SARVAM_API_KEY")
        self.key = api_key
        self.timeout = timeout
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "stt_calls": 0, "tts_chars": 0}
        # one entry per billable call: {"kind": text|image|translate|stt|tts, "model", "in", "out", "cost"}
        self.ledger: list[dict] = []
        self.budget = pricing.FREE_CREDITS

    @property
    def spent(self) -> float:
        return sum(e["cost"] for e in self.ledger)

    def _guard(self):
        if self.spent >= self.budget:
            raise SarvamError(f"Session budget of {pricing.inr(self.budget)} reached; no more API calls made.")

    def _log(self, kind, model, tin=0, tout=0, cost=0.0):
        self.ledger.append({"kind": kind, "model": model, "in": tin, "out": tout, "cost": cost})

    # ---------- low level ----------
    def _post(self, path: str, *, json_body=None, files=None, data=None, retries: int = 1):
        self._guard()
        headers = {"api-subscription-key": self.key}
        for attempt in range(retries + 1):
            r = requests.post(f"{BASE}{path}", headers=headers, json=json_body,
                              files=files, data=data, timeout=self.timeout)
            if r.status_code in (429, 503) and attempt < retries:
                time.sleep(2)
                continue
            if r.status_code >= 400:
                raise SarvamError(f"{path} -> {r.status_code}: {r.text[:300]}")
            return r.json()

    def _content(self, resp, kind: str, model: str) -> str:
        u = resp.get("usage") or {}
        tin, tout = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += tin
        self.usage["completion_tokens"] += tout
        self._log(kind, model, tin, tout, pricing.llm_cost(model, tin, tout))
        ch = resp["choices"][0]
        self.last_finish = ch.get("finish_reason")
        self.last_had_reasoning = bool(ch["message"].get("reasoning_content"))
        text = ch["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    # ---------- chat ----------
    def chat(self, messages, *, max_tokens: int = 450, temperature: float = 0.0, kind: str = "text",
             response_format: dict | None = None) -> str:
        body = {
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": None,  # thinking OFF: saves the biggest chunk of tokens
        }
        if response_format:
            body["response_format"] = response_format
        text = self._content(self._post("/v1/chat/completions", json_body=body), kind, CHAT_MODEL)
        if not text and self.last_finish == "length":
            # the whole budget went to hidden reasoning -> one retry with more room (only in this case)
            body["max_tokens"] = max_tokens * 3
            text = self._content(self._post("/v1/chat/completions", json_body=body), kind, CHAT_MODEL)
        return text

    def chat_structured(self, messages, schema: dict, name: str, max_tokens: int = 900) -> dict:
        """JSON-schema constrained output. Never raises on bad JSON: repairs unescaped quotes and
        truncated output, and drops a half-written last item if the reply was cut off."""
        fmt = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        try:
            raw = self.chat(messages, max_tokens=max_tokens, response_format=fmt)
        except SarvamError as e:
            if "400" not in str(e):
                raise
            raw = self.chat(messages, max_tokens=max_tokens, response_format={"type": "json_object"})
        self.last_raw = raw
        data = parse_json_loose(raw)
        if isinstance(data, str):                 # double-encoded JSON string
            data = parse_json_loose(data)
        if isinstance(data, list):                # bare list of parts
            data = {"parts": data}
        if self.last_finish == "length" and isinstance(data, dict):
            _drop_half_items(data)                # cut-off reply: only drop half-written CITED items
        return data if isinstance(data, dict) else {}

    def chat_json(self, messages, max_tokens: int = 160) -> dict:
        """Small JSON call (query planning). Reasoning off; repairs bad JSON; never raises on parse."""
        raw = self.chat(messages, max_tokens=max_tokens, kind="plan", response_format={"type": "json_object"})
        data = parse_json_loose(raw)
        return data if isinstance(data, dict) else {}

    def to_english_keywords(self, text: str) -> str:
        """Only used for typed non-English questions (voice uses Saaras translate instead)."""
        return self.chat([
            {"role": "system", "content": "Translate to English. Output only the translation."},
            {"role": "user", "content": text[:400]},
        ], max_tokens=80, kind="translate")

    # ---------- vision ----------
    def describe_image(self, image_bytes: bytes) -> str:
        small, mime = shrink_image(image_bytes)
        b64 = base64.b64encode(small).decode()
        body = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Motorcycle problem photo. In 1-2 short sentences state only what is visible "
                                         "(part, smoke/fluid colour, warning symbols, damage). No causes, no advice."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "temperature": 0,
            "max_tokens": 80,
        }
        return self._content(self._post("/v2/chat/completions", json_body=body), "image", VISION_MODEL)

    # ---------- speech ----------
    def transcribe_to_english(self, audio_bytes: bytes, filename: str = "q.wav") -> tuple[str, str | None]:
        """Saaras translate mode: any Indian language speech -> English text + detected language."""
        self.usage["stt_calls"] += 1
        resp = self._post(
            "/speech-to-text",
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": "saaras:v3", "mode": "translate"},
        )
        secs = _wav_seconds(audio_bytes)
        self._log("stt", "saaras:v3", cost=pricing.stt_cost(secs))
        return resp.get("transcript", ""), resp.get("language_code")

    def speak(self, text: str, language_code: str = "en-IN") -> bytes:
        text = re.sub(r"\[p\.\s*\d+[^\]]*\]", "", text)
        text = re.sub(r"[*#_`>]", "", text)[:1200]
        self.usage["tts_chars"] += len(text)
        resp = self._post("/text-to-speech", json_body={
            "text": text, "language_code": language_code,
            "model": "bulbul:v3", "speaker": "shubh",
        })
        self._log("tts", "bulbul:v3", cost=pricing.tts_cost(len(text)))
        return base64.b64decode("".join(resp["audios"]))
