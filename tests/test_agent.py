import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag import load_pdf, Index
import agent, pricing
from sarvam_client import Sarvam

HERE = os.path.dirname(__file__)
PDF = os.path.join(HERE, "sample_manual.pdf")


class FakeSarvam:
    """Stand-in for Sarvam. Records calls. The planner mimics the real model: splits multi-questions
    and resolves pronouns from the conversation; the answerer returns one part per question."""
    def __init__(self, ans=None, plan=None):
        self.ans, self.plan_override, self.calls, self.last, self.plan_input = ans, plan, [], None, None

    def describe_image(self, b):
        self.calls.append("image"); return "Thick white smoke coming out of the exhaust pipe."

    def chat_json(self, msgs, **kw):
        self.calls.append("plan"); self.plan_input = msgs[-1]["content"]
        if self.plan_override:
            return self.plan_override
        latest = re.search(r"LATEST MESSAGE: (.*)", self.plan_input).group(1)
        if "बाइक" in latest:
            return {"standalone": "Engine does not start", "queries": ["engine does not start"], "language": "hi-IN"}
        if " and " in latest:
            a, b = latest.split(" and ", 1)
            return {"standalone": latest, "queries": [a.strip(" ?"), b.strip(" ?")]}
        if re.search(r"\bit\b", latest) and "smoke" in self.plan_input:
            return {"standalone": "What to do if white smoke from the exhaust keeps happening?",
                    "queries": ["white smoke exhaust persists"]}
        return {"standalone": latest, "queries": [latest]}

    def chat_structured(self, msgs, schema, name, **kw):
        self.calls.append("chat"); self.last = msgs
        if self.ans == "NOT_IN_MANUAL":
            return {"parts": [{"question": "", "found": False, "summary": "", "spec": [], "steps": [],
                               "service_centre": [], "warnings": []}], "not_covered": ""}
        pages = re.findall(r"\[p\.(\d+)\]", msgs[-1]["content"])
        qs = re.findall(r"^\d+\. (.*)$", msgs[-1]["content"], re.M)
        return {"parts": [{"question": q, "found": True, "summary": f"Answer to {q}",
                           "spec": [], "steps": [{"text": "Check the engine oil level", "page": pages[0]},
                                                 {"text": "Bogus step", "page": "99"}],
                           "service_centre": [], "warnings": []} for q in qs], "not_covered": ""}


def idx():
    if not os.path.exists(PDF):
        os.system(f"{sys.executable} {HERE}/make_sample_manual.py {PDF}")
    return Index(load_pdf(open(PDF, "rb").read())[0])


def turn(f, ix, q, h, **kw):
    r = agent.answer(f, ix, q, h, **kw)
    h += [{"role": "user", "content": q, "standalone": r.standalone, "img_desc": r.image_description},
          {"role": "assistant", "content": r.answer, "summary": r.summary}]
    return r


# ---------- call budget ----------
def test_simple_first_question_is_one_call():
    f = FakeSarvam(); r = agent.answer(f, idx(), "Why is white smoke coming from my bike?", [])
    assert r.found and f.calls == ["chat"] and r.api_calls == 1

def test_image_adds_one_call_and_is_cached():
    f, cache = FakeSarvam(), {}
    agent.answer(f, idx(), "Why is this smoke coming?", [], image=b"img", image_desc_cache=cache)
    agent.answer(f, idx(), "Why is this smoke coming?", [], image=b"img", image_desc_cache=cache)
    assert f.calls.count("image") == 1

def test_out_of_scope_refused_without_an_answer_call():
    """Weak first question -> one small planner call to reword it; still nothing -> refuse, no answer call."""
    f = FakeSarvam(); r = agent.answer(f, idx(), "How do I pair bluetooth with my phone app?", [])
    assert not r.found and f.calls == ["plan"] and r.reason.startswith("retrieval")

def test_offline_out_of_scope_zero_calls():
    r = agent.answer(None, idx(), "How do I pair bluetooth with my phone app?", [])
    assert not r.found and r.api_calls == 0

def test_weak_first_question_is_reworded_by_planner():
    """'how much petrol does the tank hold'-style gap: planner rewords, stronger retrieval wins."""
    class Reword(FakeSarvam):
        def chat_json(self, *a, **k):
            self.calls.append("plan"); return {"standalone": "What causes white smoke from the exhaust?", "queries": ["white smoke exhaust"]}
    f = Reword(); r = agent.answer(f, idx(), "why is there vapour coming out the back?", [])
    assert f.calls[0] == "plan" and r.queries == ["white smoke exhaust"] and r.found

def test_offline_mode_zero_calls():
    r = agent.answer(None, idx(), "engine does not start", [])
    assert r.found and r.api_calls == 0 and "p.2" in r.answer

# ---------- multi-question ----------
def test_multi_question_retrieves_per_part_and_answers_each():
    f = FakeSarvam(); r = agent.answer(f, idx(), "Why is there white smoke and how do I adjust the clutch?", [])
    assert f.calls == ["plan", "chat"] and len(r.queries) == 2
    prompt = f.last[-1]["content"]
    assert "(for: Why is there white smoke)" in prompt and "(for: how do I adjust the clutch)" in prompt
    assert "#### 1." in r.answer and "#### 2." in r.answer

# ---------- multi-turn ----------
def test_followup_is_resolved_by_planner_with_memory():
    f, ix, h = FakeSarvam(), idx(), []
    turn(f, ix, "Why is white smoke coming from my bike?", h)
    r2 = turn(f, ix, "what should I do if it keeps happening?", h)
    assert "CONVERSATION:" in f.plan_input and "white smoke" in f.plan_input.lower()
    assert r2.followup and "smoke" in r2.standalone.lower() and r2.found

def test_topic_does_not_accumulate_or_stick():
    """Regression: turn 3 on a new topic must not search old topics or reuse old pages."""
    plans = iter([
        {"standalone": "How do I adjust clutch free play?", "queries": ["clutch free play adjustment"]},
        {"standalone": "How do I check the engine oil level?", "queries": ["engine oil level check"]},
    ])
    f, ix, h = FakeSarvam(), idx(), []
    turn(f, ix, "How do I adjust clutch free play?", h)
    f.plan_override = next(plans)
    turn(f, ix, "clutch again?", h)
    f.plan_override = next(plans)
    r3 = turn(f, ix, "How do I check the engine oil level?", h)
    assert r3.queries == ["engine oil level check"] and "clutch" not in r3.query.lower().split("[+")[0]
    assert "(for: engine oil level check)" in f.last[-1]["content"]

def test_memory_uses_standalone_questions_not_growing_chain():
    h = [{"role": "user", "content": "and the chain?", "standalone": "How much drive chain slack?"},
         {"role": "assistant", "content": "long answer...", "summary": "20-30 mm slack"}]
    mem = agent._memory(h)
    assert "How much drive chain slack?" in mem and "20-30 mm" in mem and "long answer" not in mem

def test_context_changes_cache_key():
    h1 = [{"role": "user", "content": "q", "standalone": "tyre pressure?"}, {"role": "assistant", "content": "a"}]
    h2 = [{"role": "user", "content": "q", "standalone": "engine oil?"}, {"role": "assistant", "content": "a"}]
    assert agent.context_signature("how often?", h1) != agent.context_signature("how often?", h2)
    assert agent.context_signature("how often?", []) == ""

def test_planner_failure_falls_back_locally():
    class Broken(FakeSarvam):
        def chat_json(self, *a, **k): raise RuntimeError("503")
    f, h = Broken(), [{"role": "user", "content": "white smoke?", "standalone": "white smoke"},
                      {"role": "assistant", "content": "x", "summary": "y"}]
    r = agent.answer(f, idx(), "what if it continues?", h)
    assert r.found and "white smoke" in r.standalone

# ---------- grounding & robustness ----------
def test_citation_filter():
    r = agent.answer(FakeSarvam(), idx(), "white smoke from exhaust", [])
    assert "p.99" not in r.answer and "Bogus" not in r.answer and "1. Check the engine oil level" in r.answer

def test_model_says_not_in_manual():
    r = agent.answer(FakeSarvam(ans="NOT_IN_MANUAL"), idx(), "why smoke", [])
    assert not r.found

def test_hindi_typed_is_planned_and_answered_in_hindi():
    f = FakeSarvam(); r = agent.answer(f, idx(), "बाइक स्टार्ट नहीं हो रही", [])
    assert r.language == "hi-IN" and f.calls == ["plan", "chat"]

def test_voice_path_first_question_skips_planner():
    f = FakeSarvam(); agent.answer(f, idx(), "engine not starting", [], english_query="engine not starting", lang="hi-IN")
    assert f.calls == ["plan", "chat"] or f.calls == ["chat"]

def test_photo_failed_gives_actionable_message():
    class NoVision(FakeSarvam):
        def describe_image(self, b): raise RuntimeError("400 beta access")
    r = agent.answer(NoVision(), idx(), "What is wrong in this photo and what should I do?", [], image=b"x")
    assert r.reason

def test_page_resolution_variants():
    allowed = {"81", "82"}
    srcs = [("82", "Loosen the cover end adjuster nuts at cover end completely.")]
    assert agent.resolve_page("p.82", "x", allowed, srcs, {}) == "82"
    assert agent.resolve_page("[p.81/82]", "x", allowed, srcs, {}) == "81"
    assert agent.resolve_page("84", "x", allowed, srcs, {"84": "82"}) == "82"
    assert agent.resolve_page("", "Loosen the cover end adjuster nuts", allowed, srcs, {}) == "82"
    assert agent.resolve_page("999", "Replace the spark plug", allowed, srcs, {}) is None

def test_render_accepts_old_single_shape():
    md, kept, _ = agent.render({"summary": "S", "steps": [{"text": "Do X", "page": "2"}]}, "en-IN", {"2"})
    assert kept == 1 and "Do X" in md

def test_stemming_consistent():
    from rag import tokenize
    assert tokenize("brakes") == tokenize("brake") == tokenize("braking")
    assert tokenize("smoke") == tokenize("smoking")

# ---------- client ----------
def test_cost_math_and_budget_guard():
    c = Sarvam("k"); c._log("image", "gemma4", 300, 40, pricing.llm_cost("gemma4", 300, 40))
    assert abs(c.spent - (300*36.6 + 40*91.5)/1e6) < 1e-12
    c.budget = 0.0
    try: c._post("/x"); assert False
    except Exception as e: assert "budget" in str(e)

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

def test_bad_json_is_repaired_not_raised():
    from sarvam_client import parse_json_loose
    broken = '{"found": true, "summary": "Turn the "OFF" switch", "steps": [{"text": "Loosen nuts", "page": "82"}, {"text": "Tigh'
    d = parse_json_loose(broken)
    assert d["found"] is True and d["steps"][0]["page"] == "82"


# ---------- regressions from the live eval run ----------
def _mock_reply(content, finish="stop"):
    class R:
        status_code = 200; text = ""
        def json(s): return {"choices": [{"finish_reason": finish, "message": {"content": content}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
    return R()

def test_truncated_single_part_reply_is_not_deleted():
    """Root cause of the live refusals: cut-off replies had their only 'part' popped."""
    import sarvam_client as sc
    from unittest import mock
    cut = ('{"parts":[{"question":"clutch","found":true,"summary":"Set 10-12 mm.","spec":[{"text":"10-12 mm","page":"81"}],'
           '"steps":[{"text":"Loosen the cover end adjuster nuts","page":"82"},{"text":"Tighten the adj')
    with mock.patch("requests.post", lambda *a, **k: _mock_reply(cut, "length")):
        d = sc.Sarvam("k").chat_structured([{"role": "user", "content": "q"}], {}, "n")
    assert len(d["parts"]) == 1 and [x["text"] for x in d["parts"][0]["steps"]] == ["Loosen the cover end adjuster nuts"]

def test_normalise_variants():
    assert agent._normalise({"answers": [{"summary": "x", "steps": [{"text": "a", "page": "1"}], "found": "false"}]})["parts"][0]["found"]
    assert not agent._normalise({"parts": [{"found": False, "summary": "", "steps": []}]})["parts"][0]["found"]
    assert agent._normalise({"summary": "s", "steps": [{"text": "a", "page": "1"}]})["parts"][0]["found"]

def test_double_encoded_and_list_json():
    import sarvam_client as sc, json
    from unittest import mock
    inner = json.dumps({"parts": [{"question": "q", "found": True, "summary": "s", "steps": [{"text": "a", "page": "1"}]}]})
    for content in (json.dumps(inner), json.dumps([{"question": "q", "found": True, "summary": "s"}])):
        with mock.patch("requests.post", lambda *a, **k: _mock_reply(content)):
            d = sc.Sarvam("k").chat_structured([{"role": "user", "content": "q"}], {}, "n")
        assert d["parts"] and d["parts"][0]["question"] == "q"

def test_refusal_with_strong_match_shows_manual_text():
    f = FakeSarvam(ans="NOT_IN_MANUAL")
    agent.CONFIDENT_SCORE, old = 1.0, agent.CONFIDENT_SCORE
    try:
        r = agent.answer(f, idx(), "white smoke from the exhaust", [])
    finally:
        agent.CONFIDENT_SCORE = old
    assert r.found and "Here is what the manual says" in r.answer and "p.1" in r.answer and "showing the manual" in r.reason

def test_uncited_summary_with_wrong_numbers_is_dropped():
    srcs = [("71", "Tyre pressure Front Rear Solo 32 psi 32 psi With Pillion 32 psi 36 psi")]
    md, _, _ = agent.render({"parts": [{"question": "tyre", "found": True, "summary": "Tyre pressure is 40 psi"}]},
                            "en-IN", {"71"}, srcs)
    assert "40 psi" not in md
    md, _, _ = agent.render({"parts": [{"question": "tyre", "found": True, "summary": "Tyre pressure solo is 32 psi"}]},
                            "en-IN", {"71"}, srcs)
    assert "32 psi" in md


# ---------- maintenance chart + duplicates ----------
def test_maintenance_chart_rows_rebuilt_from_columns():
    from rag import normalize_maintenance
    lines = ["PERIODICAL MAINTENANCE", "km (x 1,000)", "0.5", "5", "10", "Months", "1.5", "6", "12",
             "1", "Engine oil (Level check/replace)", "R", "I", "R", "Check level at every 1,000 km",
             "2", "Engine oil filter element", "R", "R"]
    st = {"grid": {2: ["R", "", "R"]}}
    out = normalize_maintenance(lines, st)
    row1 = next(l for l in out if l.startswith("Maintenance item 1."))
    assert "Replace at 0.5, 10 thousand km (1.5, 12 months)" in row1 and "Inspect at 5 thousand km" in row1
    assert "Check level at every 1,000 km" in row1                   # note kept, not merged into interval
    row2 = next(l for l in out if l.startswith("Maintenance item 2."))
    assert "Replace at 0.5, 10 thousand km" in row2                   # blank cell honoured via grid

def test_duplicate_lines_removed():
    md, kept, _ = agent.render({"parts": [{"question": "tyre", "found": True, "summary": "s",
        "spec": [{"text": "Rear 36 psi with pillion", "page": "71"}, {"text": "Rear 36 psi with pillion", "page": "104"}],
        "steps": [{"text": "Contact Authorised service center", "page": "108"}],
        "service_centre": [{"text": "Contact authorised service center.", "page": "109"}], "warnings": []}]},
        "en-IN", {"71", "104", "108", "109"})
    assert md.count("36 psi") == 1 and md.lower().count("contact authorised service center") == 1


# ---------- answer tidiness (from live eval output) ----------
def test_spec_guard_moves_non_specs_and_keeps_advice_first():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "Keep positive free play and visit a service centre.",
          "spec": [P("Keep positive free play", "82"), P("Visit nearest service centre", "82"), P("Free play 10-12 mm", "82")],
          "steps": [P(f"Step {i}", "82") for i in range(1, 12)], "service_centre": [], "warnings": []}]}
    md, _, _ = agent.render(d, "en-IN", {"82"})
    spec_block = md.split("**📏")[1].split("**🔧")[0]
    assert "10-12 mm" in spec_block and "Keep positive" not in spec_block and "Visit nearest" not in spec_block
    assert "1. Keep positive free play" in md and "remaining steps are in the manual" in md
    assert "Visit nearest service centre" in md.split("**🏪")[1]

def test_near_duplicate_specs_collapse_but_distinct_values_survive():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "s", "steps": [], "service_centre": [], "warnings": [],
          "spec": [P("Front 32 psi", "16"), P("Rear 36 psi", "16"), P("Front 32 psi, Rear 36 psi", "71"),
                   P("Rear 32 psi solo", "71")]}]}
    md, _, _ = agent.render(d, "en-IN", {"16", "71"})
    assert md.count("36 psi") == 1 and "Rear 32 psi solo" in md

def test_lazy_planner_echo_still_carries_context():
    class Lazy(FakeSarvam):
        def chat_json(self, *a, **k):
            self.calls.append("plan"); return {"standalone": "What should I do if it keeps happening?", "queries": ["keeps happening"]}
    h = [{"role": "user", "content": "white smoke from exhaust", "standalone": "white smoke from exhaust"},
         {"role": "assistant", "content": "x", "summary": "condensation"}]
    r = agent.answer(Lazy(), idx(), "What should I do if it keeps happening?", h)
    assert "white smoke" in r.queries[0] and "following up on" in r.standalone


# ---------- regressions from live eval run #4 ----------
def test_connective_start_is_not_a_dangling_followup():
    """'And the engine oil grade?' names its own subject: don't glue the tyre question onto it."""
    class Echo(FakeSarvam):
        def chat_json(self, *a, **k):
            self.calls.append("plan"); return {"standalone": "What is the engine oil grade?", "queries": ["engine oil grade"]}
    h = [{"role": "user", "content": "is it the same with a pillion?", "standalone": "What is the tyre pressure with a pillion?"},
         {"role": "assistant", "content": "x", "summary": "36 psi"}]
    r = agent.answer(Echo(), idx(), "And the engine oil grade?", h)
    assert r.queries == ["engine oil grade"] and "tyre" not in r.standalone.lower()

def test_two_symptoms_joined_by_and_are_split():
    class Merge(FakeSarvam):
        def chat_json(self, *a, **k):
            self.calls.append("plan"); return {"standalone": "ABS light on and bike won't start", "queries": ["ABS light on engine does not start"]}
    r = agent.answer(Merge(), idx(), "My ABS light is on and the bike won't start", [])
    assert r.queries == ["My ABS light is on", "the bike won't start"]

def test_info_questions_get_no_steps_and_offtopic_warnings_dropped():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "engine oil grade", "found": True, "summary": "Use SAE 15W 50.", "_info": True,
          "_q": "engine oil grade",
          "spec": [P("SAE 15W 50 API SL", "2"), P("Endurance Gabriel Semi Synthetic", "2")],
          "steps": [P("Check oil level on center stand", "3")], "service_centre": [],
          "warnings": [P("Wrong oil grade reduces engine life", "2"),
                       P("Do not attempt to shift gears without moving back and forth", "2")]}]}
    md, _, _ = agent.render(d, "en-IN", {"2", "3"})
    assert "🔧" not in md and "Endurance Gabriel Semi Synthetic" in md.split("**📏")[1]
    assert "oil grade reduces" in md and "back and forth" not in md

def test_duplicate_by_meaning_removed():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "fuse", "found": True, "summary": "s", "spec": [], "service_centre": [], "warnings": [],
          "steps": [P("Replace fuse with same rating", "8"), P("Open side panel", "9"), P("Replace the fuse with the same rating", "8")]}]}
    md, _, _ = agent.render(d, "en-IN", {"8", "9"})
    assert md.lower().count("same rating") == 1

def test_steps_differing_only_by_number_are_kept():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "s", "spec": [], "service_centre": [], "warnings": [],
          "steps": [P("Tighten nut to 10 Nm", "8"), P("Tighten nut to 20 Nm", "8")]}]}
    md, _, _ = agent.render(d, "en-IN", {"8"})
    assert "10 Nm" in md and "20 Nm" in md

def test_spec_consequences_and_actions_move_out():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "s", "service_centre": [], "warnings": [],
          "spec": [P("Chain slackness beyond 30 mm will lead to chain slippage", "8"), P("Unwind hand adjuster 4 full turns", "8"),
                   P("25 - 30 mm", "8"), P("Replace at 0.5, 10 thousand km", "8")],
          "steps": [P("Replace with same rating", "8"), P("Replace fuse with same rating", "8")]}]}
    md, _, _ = agent.render(d, "en-IN", {"8"})
    spec = md.split("**📏")[1].split("**🔧")[0]
    assert "25 - 30 mm" in spec and "Replace at 0.5" in spec and "will lead" not in spec and "Unwind" not in spec
    assert "will lead" in md.split("**⚠️")[1] and md.lower().count("same rating") == 1


# ---------- regressions from a real user conversation on another manual ----------
def test_contractions_are_normalised():
    from rag import tokenize
    assert tokenize("lights don't work") == ["light", "work"]
    assert tokenize("Don’t drink and ride") == ["drink", "rid"]           # no stray 'don' / 't' tokens
    assert "not" not in tokenize("won't start") and "start" in tokenize("won't start")

def test_topic_bleed_guard():
    h = [{"role": "user", "content": "how to put petrol", "standalone": "How do I refuel?"},
         {"role": "assistant", "content": "x", "summary": "Use unleaded petrol"}]
    assert agent.guard_topic_bleed("no my bike lights don't work",
                                   ["fuel tank capacity", "lights do not work", "petrol tank cap open"], h, "en-IN") == ["lights do not work"]
    assert agent.guard_topic_bleed("is it the same with a pillion?", ["tyre pressure with pillion"], h, "en-IN") == ["tyre pressure with pillion"]
    assert agent.guard_topic_bleed("no petrol", ["refuelling fuel tank empty"], h, "en-IN") == ["refuelling fuel tank empty"]
    assert agent.guard_topic_bleed("lights don't work", ["fuel tank capacity"], h, "en-IN") == ["lights don't work"]   # all bled -> message itself

def test_near_duplicate_steps_merge_but_numbers_protect():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "s", "spec": [], "service_centre": [], "warnings": [],
          "steps": [P("Wait for 10 seconds before trying again", "2"), P("Wait 10 seconds before repeating", "2"),
                    P("Wait 20 seconds before repeating", "2")]}]}
    md, _, _ = agent.render(d, "en-IN", {"2"})
    assert md.count("10 seconds") == 1 and "20 seconds" in md


# ---------- references at the end + answers in the user's language ----------
def test_references_listed_once_at_the_end():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "Set free play to 10-12 mm.",
          "spec": [P("Free play 10-12 mm", "81")], "steps": [P("Loosen the adjuster nuts", "82")],
          "service_centre": [], "warnings": []}]}
    md, _, _ = agent.render(d, "en-IN", {"81", "82"})
    assert "(p." not in md and md.strip().splitlines()[-1] == "📖 **Sources:** p.81, p.82"
    assert agent.cited_pages(md) == {"81", "82"}

class FakeTranslator:
    """Stand-in for Sarvam Translate: prefixes each line with [hi] so we can see what was translated."""
    def __init__(self, merge=False, fail=False): self.merge, self.fail, self.calls = merge, fail, []
    def translate(self, text, target, source="en-IN"):
        self.calls.append(text)
        if self.fail: raise RuntimeError("503")
        lines = [f"[{target[:2]}] {l}" for l in text.split("\n")]
        return " ".join(lines) if (self.merge and len(lines) > 1) else "\n".join(lines)

def _sample_md():
    P = lambda t, p: {"text": t, "page": p}
    d = {"parts": [{"question": "q", "found": True, "summary": "Check the kill switch and fuel.",
          "spec": [P("Battery 12 V 3 Ah", "99")], "steps": [P("Switch ON the ignition", "74"), P("Top up the fuel", "74")],
          "service_centre": [P("Contact your dealer if the lamp stays on", "74")], "warnings": []}]}
    return agent.render(d, "en-IN", {"74", "99"})[0]

def test_answer_translated_with_layout_kept():
    t = FakeTranslator()
    out, ok = agent.translate_answer(t, _sample_md(), "hi-IN")
    assert ok and len(t.calls) == 1                                  # batched into one request
    assert "**[hi] Check the kill switch and fuel.**" in out            # bold kept
    assert "1. [hi] Switch ON the ignition" in out and "- [hi] Battery 12 V 3 Ah" in out
    assert "**📏 [hi] Specification**" in out                          # headings translated, emoji kept
    assert out.strip().splitlines()[-1] == "📖 **स्रोत:** p.74, p.99"   # sources kept, label localised

def test_translation_falls_back_line_by_line_then_to_english():
    t = FakeTranslator(merge=True)
    out, ok = agent.translate_answer(t, _sample_md(), "kn-IN")
    assert ok and "1. [kn] Switch ON the ignition" in out and len(t.calls) > 1
    md = _sample_md()
    out, ok = agent.translate_answer(FakeTranslator(fail=True), md, "hi-IN")
    assert not ok and out == md                                        # English shown, never an error

def test_english_answers_are_not_translated():
    t = FakeTranslator(); md = _sample_md()
    assert agent.translate_answer(t, md, "en-IN") == (md, True) and t.calls == []

def test_hindi_question_gets_hindi_answer_end_to_end():
    class Model(FakeSarvam, FakeTranslator):
        def __init__(self): FakeSarvam.__init__(self); FakeTranslator.__init__(self)
    m = Model()
    r = agent.answer(m, idx(), "बाइक स्टार्ट नहीं हो रही", [])
    assert r.language == "hi-IN" and "[hi]" in r.answer and "स्रोत" in r.answer
    assert "Write all text in English" in m.last[0]["content"]          # model writes English, app translates
