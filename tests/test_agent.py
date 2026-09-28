import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag import load_pdf, Index
import agent, pricing
from sarvam_client import Sarvam

HERE = os.path.dirname(__file__)
PDF = os.path.join(HERE, "sample_manual.pdf")

class FakeSarvam:
    """Counts calls; mimics answer that cites the first excerpt page + a bogus page."""
    def __init__(self, ans=None): self.ans = ans; self.calls = []
    def describe_image(self, b): self.calls.append("image"); return "Thick white smoke coming out of the exhaust pipe."
    def to_english_keywords(self, t): self.calls.append("translate"); return "bike not starting"
    def chat_structured(self, msgs, schema, name, **kw):
        self.calls.append("chat"); self.last = msgs
        if self.ans == "NOT_IN_MANUAL":
            return {"found": False, "summary": "", "spec": [], "steps": [], "service_centre": [], "warnings": [], "not_covered": ""}
        pg = re.search(r"\[p\.(\d+)", msgs[-1]["content"]).group(1)
        return {"found": True, "summary": "White smoke when cold is condensation.",
                "spec": [], "steps": [{"text": "Check the engine oil level", "page": pg},
                                      {"text": "Bogus step", "page": "99"}],
                "service_centre": [{"text": "If smoke persists", "page": pg}], "warnings": [], "not_covered": ""}

def idx():
    if not os.path.exists(PDF):
        os.system(f"{sys.executable} {HERE}/make_sample_manual.py {PDF}")
    return Index(load_pdf(open(PDF, "rb").read())[0])

def test_text_question_is_one_call():
    f = FakeSarvam(); r = agent.answer(f, idx(), "Why is white smoke coming from my bike?", [])
    assert r.found and f.calls == ["chat"] and r.api_calls == 1

def test_image_adds_one_call_and_is_cached():
    f, cache = FakeSarvam(), {}
    agent.answer(f, idx(), "Why is this smoke coming?", [], image=b"img", image_desc_cache=cache)
    agent.answer(f, idx(), "and what should I check?", [], image=b"img", image_desc_cache=cache)
    assert f.calls.count("image") == 1

def test_out_of_scope_refused_locally_zero_calls():
    f = FakeSarvam(); r = agent.answer(f, idx(), "How do I pair bluetooth with my phone app?", [])
    assert not r.found and f.calls == []

def test_citation_filter():
    r = agent.answer(FakeSarvam(), idx(), "white smoke from exhaust", [])
    assert "p.99" not in r.answer and "Bogus" not in r.answer and "1. Check the engine oil level" in r.answer

def test_offline_mode_zero_calls():
    r = agent.answer(None, idx(), "engine does not start", [])
    assert r.found and r.api_calls == 0 and "p.2" in r.answer

def test_hindi_typed_translates_then_answers():
    f = FakeSarvam(); r = agent.answer(f, idx(), "बाइक स्टार्ट नहीं हो रही", [])
    assert r.language == "hi-IN" and f.calls == ["translate", "chat"]

def test_voice_path_skips_translate():
    f = FakeSarvam(); agent.answer(f, idx(), "engine not starting", [], english_query="engine not starting", lang="hi-IN")
    assert f.calls == ["chat"]

def test_cost_math_and_budget_guard():
    c = Sarvam("k"); c._log("image", "gemma4", 300, 40, pricing.llm_cost("gemma4", 300, 40))
    assert abs(c.spent - (300*36.6 + 40*91.5)/1e6) < 1e-12
    c.budget = 0.0
    try: c._post("/x"); assert False
    except Exception as e: assert "budget" in str(e)

def test_stemming_consistent():
    from rag import tokenize
    assert tokenize("brakes") == tokenize("brake") == tokenize("braking")
    assert tokenize("smoke") == tokenize("smoking")

def test_refusal_has_reason():
    r = agent.answer(FakeSarvam(), idx(), "How do I pair bluetooth with my phone app?", [])
    assert r.reason.startswith("retrieval")

def test_photo_failed_gives_actionable_message():
    class NoVision(FakeSarvam):
        def describe_image(self, b): raise RuntimeError("400 beta access")
    r = agent.answer(NoVision(), idx(), "What is wrong in this photo and what should I do?", [], image=b"x")
    assert "photo" in r.answer and r.reason

def test_empty_content_retry():
    import sarvam_client as sc
    from unittest import mock
    calls = []
    class R:
        status_code = 200; text = ""
        def __init__(s, d): s.d = d
        def json(s): return s.d
    def fake(url, **kw):
        calls.append(kw["json"]["max_tokens"])
        if len(calls) == 1:
            return R({"choices":[{"finish_reason":"length","message":{"content":None,"reasoning_content":"..."}}],"usage":{}})
        return R({"choices":[{"finish_reason":"stop","message":{"content":"ok [p.1]"}}],"usage":{}})
    with mock.patch("requests.post", fake):
        out = sc.Sarvam("k").chat([{"role":"user","content":"x"}])
    assert out == "ok [p.1]" and calls == [450, 1350]

def test_page_resolution_variants():
    allowed = {"81", "82"}
    srcs = [("82", "Loosen the cover end adjuster nuts at cover end completely.")]
    assert agent.resolve_page("p.82", "x", allowed, srcs, {}) == "82"
    assert agent.resolve_page("[p.81/82]", "x", allowed, srcs, {}) == "81"
    assert agent.resolve_page("84", "x", allowed, srcs, {"84": "82"}) == "82"          # PDF index
    assert agent.resolve_page("", "Loosen the cover end adjuster nuts", allowed, srcs, {}) == "82"  # by wording
    assert agent.resolve_page("999", "Replace the spark plug", allowed, srcs, {}) is None

def test_short_question_not_polluted_by_history():
    f = FakeSarvam()
    r = agent.answer(f, idx(), "clutch adjust", [{"role": "user", "content": "brake issues"}])
    assert "brake" not in r.query.split("[+")[0]

def test_bad_json_is_repaired_not_raised():
    from sarvam_client import parse_json_loose
    broken = '{"found": true, "summary": "Turn the "OFF" switch", "steps": [{"text": "Loosen nuts", "page": "82"}, {"text": "Tigh'
    d = parse_json_loose(broken)
    assert d["found"] is True and d["steps"][0]["page"] == "82"
    assert parse_json_loose("") == {} and not isinstance(parse_json_loose("not json at all"), dict) or parse_json_loose("not json at all") == {}

def test_truncated_reply_drops_half_item():
    import sarvam_client as sc
    from unittest import mock
    class R:
        status_code = 200; text = ""
        def json(s): return {"choices": [{"finish_reason": "length", "message": {"content":
            '{"found": true, "summary": "x", "steps": [{"text": "Loosen nuts", "page": "82"}, {"text": "Tigh'}}], "usage": {}}
    with mock.patch("requests.post", lambda *a, **k: R()):
        d = sc.Sarvam("k").chat_structured([{"role": "user", "content": "q"}], {}, "n")
    assert [s["text"] for s in d["steps"]] == ["Loosen nuts"]
