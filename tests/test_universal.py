"""The agent must work on ANY bike manual, not just the Royal Enfield one it was developed with.
These tests use a synthetic manual for a fictional 'Acme Roadster 150' with a deliberately different
layout (Title Case bold headings, page numbers at the bottom, 'Trouble Shooting', a 750/3,000/6,000 km
schedule table with its own legend) and check that nothing brand-specific is needed."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import agent
from rag import load_pdf, Index

HERE = os.path.dirname(__file__)
PDF = os.path.join(HERE, "other_manual.pdf")


def other():
    if not os.path.exists(PDF):
        os.system(f"{sys.executable} {HERE}/make_other_manual.py {PDF}")
    chunks, stats = load_pdf(open(PDF, "rb").read())
    return chunks, stats, Index(chunks)


def test_layout_is_detected_not_assumed():
    chunks, stats, _ = other()
    assert "big" in stats["heading_style"]                       # Title Case headings via font size
    heads = {c.heading for c in chunks}
    assert {"Tyre Pressure", "Checking Engine Oil", "Trouble Shooting", "Periodic Maintenance Schedule"} <= heads
    assert {c.label for c in chunks} == {"1", "2", "3", "4", "5", "6"}  # page numbers printed at the BOTTOM
    assert not any("Acme Roadster 150 - Owner's Manual" in c.text for c in chunks)   # running footer removed

def test_any_schedule_table_is_rebuilt_with_its_own_legend():
    chunks, _, _ = other()
    text = "\n".join(c.text for c in chunks if c.label == "5")
    assert "Engine oil: Replace at 750, 6,000, 12,000 km (1, 8, 16 months); Inspect at 3,000, 9,000 km" in text
    assert "Spark plug: Inspect at 3,000 km (4 months); Replace at 9,000 km (12 months)" in text   # blank cells kept
    assert "Drive chain: Lubricate at 750" in text
    assert "\nItem\n" not in text and "\nMonths\n" not in text      # raw table labels removed

def test_retrieval_on_a_different_manual():
    _, _, idx = other()
    cases = {"What is the tyre pressure?": "2", "How often should I change the engine oil?": "5",
             "My engine won't start": "4", "How do I check the engine oil?": "3",
             "How much fuel does the tank hold?": "2", "How heavy is the bike?": "2",
             "When should the spark plug be replaced?": "5", "What is the clutch lever play?": "3"}
    for q, page in cases.items():
        secs, _, _ = agent.retrieve(idx, [q], None)
        got = {l for s in secs for l in s["labels"]}
        assert page in got, (q, got)

def test_troubleshooting_section_found_under_other_spelling():
    _, _, idx = other()
    secs, _, _ = agent.retrieve(idx, ["My engine won't start"], None)
    assert "Trouble Shooting" in secs[0]["chunks"][0].heading

def test_no_brand_specific_strings_in_prompts_or_vocabulary():
    import rag
    blob = (agent.PLAN_SYS + agent.ANSWER_SYS + str(rag.SYNONYMS)).lower()
    for brand_term in ("royal enfield", "classic 350", "tripper", "32 psi", "36 psi", "15w 50", "15w-50"):
        assert brand_term not in blob, brand_term

def test_end_to_end_interval_answer_on_other_manual():
    """Full pipeline with a stand-in model: the schedule row for engine oil reaches the model."""
    _, _, idx = other()

    class Model:
        def chat_json(self, *a, **k): return {"standalone": "How often should I change the engine oil?", "queries": ["engine oil change interval"]}
        def chat_structured(self, msgs, *a, **k):
            self.prompt = msgs[-1]["content"]
            return {"parts": [{"question": "q", "found": True, "summary": "Replace at 750, 6,000 and 12,000 km.",
                               "spec": [{"text": "Replace at 750, 6,000, 12,000 km (1, 8, 16 months)", "page": "5"}],
                               "steps": [], "service_centre": [], "warnings": []}], "not_covered": ""}
    m = Model()
    r = agent.answer(m, idx, "How often should I change the engine oil?", [])
    assert "Engine oil: Replace at 750, 6,000, 12,000 km" in m.prompt
    assert r.found and agent.cited_pages(r.answer) == {"5"} and "12,000" in r.answer
    assert "(p." not in r.answer                                   # no inline references


# ---------- lessons from a real second manufacturer's manual (Honda Shine 100) ----------
def test_split_grid_schedule_aligned_by_position_with_legend_from_other_page():
    chunks, _, _ = other()
    text = "\n".join(c.text for c in chunks if c.label == "6")
    assert "Brake shoes: Inspect (pre-ride check); Inspect at 6, 18 thousand km." in text   # offset codes, '×' unit
    assert "Clutch cable: Inspect (pre-ride check); Inspect at 1, 6, 12, 18 thousand km." in text
    assert "Drive chain lube: Inspect (pre-ride check); every 500 km: Inspect Lubricate." in text   # note cell
    assert "Fuel" not in text and "–" not in text           # '–' empty cells never read as legend/codes

def test_caps_part_numbers_are_not_headings_in_font_size_manuals():
    chunks, _, _ = other()
    heads = {c.heading for c in chunks}
    assert not any(h in heads for h in ("CPR8EA-9 (NGK)", "YTZ5S / GTZ5S", "BATTERY"))
    spec = next(c for c in chunks if c.heading == "Specifications")
    assert "CPR8EA-9 (NGK)" in spec.text                   # kept as content of the spec section

def test_interval_lookup_handles_both_row_formats():
    from rag import Chunk, Index
    idx = Index([Chunk(0, 1, "Schedule", "Maintenance item: Engine Oil: Replace at 1, 6, 12 thousand km."),
                 Chunk(1, 2, "Schedule", "Maintenance item 3. Spark plug: Replace at 20, 40 thousand km."),
                 Chunk(2, 3, "Engine Oil", "Check the engine oil level with the dipstick.")])
    assert agent._schedule_hit(idx, "How often should the engine oil be changed?")[0].id == 0
    assert agent._schedule_hit(idx, "When should the spark plug be replaced?")[0].id == 1

def test_value_questions_prefer_the_specifications_section():
    _, _, idx = other()
    for q in ("What tyre pressure should I run?", "How much does the bike weigh?", "How many litres does the tank take?"):
        secs, _, _ = agent.retrieve(idx, [q], None)
        assert {"2"} & set(secs[0]["labels"]), q      # specs / tyre-pressure page first
