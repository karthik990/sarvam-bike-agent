"""Voice questions: correct Sarvam request, English AND Indian-language speech, and no silent failures."""
import io, os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unittest import mock
import voice


def wav(seconds=2.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(seconds * rate))
    return buf.getvalue()


class Client:
    def __init__(self, text="", code=None, exc=None): self.text, self.code, self.exc = text, code, exc
    def transcribe(self, audio):
        if self.exc: raise self.exc
        return self.text, self.code


def test_request_uses_transcribe_mode_with_auto_language():
    import sarvam_client as sc
    sent = {}
    class R:
        status_code = 200; text = ""
        def json(s): return {"transcript": "my bike won't start", "language_code": "en-IN"}
    def fake(url, **kw):
        sent.update(url=url, data=kw["data"], files=kw["files"]); return R()
    with mock.patch("requests.post", fake):
        text, code = sc.Sarvam("k").transcribe(wav())
    assert sent["url"].endswith("/speech-to-text")
    assert sent["data"] == {"model": "saaras:v3", "mode": "transcribe", "language_code": "unknown"}
    assert text == "my bike won't start" and code == "en-IN"

def test_english_speech_becomes_english_question():
    r = voice.voice_to_question(Client("my bike won't start", "en-IN"), wav())
    assert r.question == "my bike won't start" and r.english_query == r.question and r.lang == "en-IN"
    assert r.message.startswith("🎙️ Heard:")

def test_hindi_speech_kept_in_hindi_for_the_planner():
    r = voice.voice_to_question(Client("बाइक स्टार्ट नहीं हो रही", "hi-IN"), wav())
    assert r.question == "बाइक स्टार्ट नहीं हो रही" and r.english_query is None and r.lang == "hi-IN"

def test_no_silent_failures():
    assert voice.voice_to_question(Client("", "en-IN"), wav()).level == "warning"          # heard nothing
    assert "30" in voice.voice_to_question(Client("x"), wav(seconds=45)).message          # too long
    assert voice.voice_to_question(None, wav()).level == "warning"                        # offline mode
    r = voice.voice_to_question(Client(exc=RuntimeError("/speech-to-text -> 422: bad audio")), wav())
    assert r.level == "error" and "422" in r.message
    assert voice.voice_to_question(Client("hi"), b"").level == "warning"                  # empty recording
