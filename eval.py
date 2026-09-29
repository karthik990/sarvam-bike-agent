"""Evaluation harness for multi-turn / multi-question accuracy.

Offline (free):   python eval.py manual.pdf
    Uses the ideal planner output written below, so it checks retrieval + section building:
    does each question reach the manual page(s) that actually contain the answer?

Live (spends credits, ~₹1 for the whole set):   python eval.py manual.pdf --live
    Runs the real pipeline (planner + answer on Sarvam) and prints each answer, how the
    follow-up was understood, the pages cited, and the cost. Needs SARVAM_API_KEY.

Expected pages are PRINTED page numbers of the Royal Enfield Classic 350 (BS-VI) owner's manual.
"""
import os
import re
import sys

from dotenv import load_dotenv

import agent
from rag import Index, load_pdf

# Each conversation: list of (user message, ideal planner queries, pages that must be retrieved)
CONVERSATIONS = {
    "clutch follow-ups": [
        ("How do I adjust clutch free play?", ["clutch free play adjustment"], {"81", "82"}),
        ("what if I can't get it to 10 mm?", ["clutch free play not achieved clutch slip"], {"82"}),
        ("and when should I get it checked?", ["clutch free play periodic maintenance interval"], {"114"}),
    ],
    "topic switches": [
        ("What is the tyre pressure?", ["tyre pressure"], {"71"}),
        ("is it the same with a pillion?", ["tyre pressure with pillion"], {"16", "71", "104"}),
        ("And the engine oil grade?", ["engine oil grade"], {"19"}),
        ("how often should I change it?", ["engine oil replace periodical maintenance"], {"111"}),
        ("How much chain slack is allowed?", ["drive chain slackness"], {"83", "84"}),
    ],
    "multi-question in one message": [
        ("What is the tyre pressure and which engine oil should I use?", ["tyre pressure", "engine oil grade"], {"71", "19"}),
        ("My ABS light is on and the bike won't start", ["ABS lamp continuously on", "engine does not start"], {"109", "108"}),
    ],
    "troubleshooting chain": [
        ("My bike won't start", ["engine does not start"], {"108"}),
        ("the lights are dim too", ["engine does not start lights dim weak horn battery"], {"108"}),
        ("what if a fuse is blown?", ["fuse blown replace"], {"108", "99", "100"}),
    ],
}


# Answer-quality checks for the live run (beyond "cited the right page")
QUALITY = {
    "and when should I get it checked?": {"no_steps": True, "must": ["1,000"]},
    "What is the tyre pressure?": {"no_steps": True, "must": ["32", "36"]},
    "is it the same with a pillion?": {"no_steps": True, "must": ["36"]},
    "And the engine oil grade?": {"no_steps": True, "must": ["15W"]},
    "how often should I change it?": {"must": ["10"], "forbid": ["every 1,000 km or 1.5", "here is what the manual says"]},
    "the lights are dim too": {"avoid_pages": {"27", "91", "92"}, "must": ["battery"]},
    "My ABS light is on and the bike won't start": {"must": ["ABS", "fuel"]},
    "What is the tyre pressure and which engine oil should I use?": {"no_steps": True, "must": ["36", "15W"]},
    "My bike won't start": {"avoid_text": ["back and forth"]},
}


def quality_issues(msg, answer):
    q, issues = QUALITY.get(msg, {}), []
    low = answer.lower()
    if q.get("no_steps") and "🔧" in answer:
        issues.append("procedure steps on a what/when question")
    for m in q.get("must", []):
        if m.lower() not in low:
            issues.append(f"missing '{m}'")
    for f in q.get("forbid", []) + q.get("avoid_text", []):
        if f.lower() in low:
            issues.append(f"contains '{f}'")
    bad = set(re.findall(r"\(p\.(\d+)\)", answer)) & q.get("avoid_pages", set())
    if bad:
        issues.append(f"cites off-topic pages {sorted(bad)}")
    return issues


def offline(index, only=None):
    """Checks retrieval twice: with the ideal planner queries, and with the RAW message
    (what a first question uses, since the planner is skipped when there's no history)."""
    total = ok = 0
    for name, turns in CONVERSATIONS.items():
        if only and only not in name:
            continue
        print(f"\n=== {name}")
        for i, (msg, queries, expected) in enumerate(turns):
            variants = [("planned", queries)] + ([("raw", [msg])] if i == 0 else [])
            if agent.split_on_and(msg):
                variants.append(("split", agent.split_on_and(msg)))
            for label, qs in variants:
                sections, _, _ = agent.retrieve(index, qs, None)
                got = {l for s in sections for l in s["labels"]}
                hit = bool(expected & got)
                total += 1; ok += hit
                print(f"  {'PASS' if hit else 'FAIL'}  [{label:7}] {msg[:48]:48} want {sorted(expected)} "
                      f"got {sorted(got, key=lambda x: int(x) if x.isdigit() else 0)}")
    print(f"\nRetrieval: {ok}/{total} checks reached the right page(s)")
    return ok == total


def live(index, only=None):
    from sarvam_client import Sarvam
    load_dotenv()
    client = Sarvam(os.environ["SARVAM_API_KEY"])
    passed = total = 0
    for name, turns in CONVERSATIONS.items():
        if only and only not in name:
            continue
        print(f"\n=== {name}")
        history = []
        for msg, _, expected in turns:
            r = agent.answer(client, index, msg, history)
            cited = set(re.findall(r"\(p\.(\d+)\)", r.answer)) | set(re.findall(r"\*\*p\.([\d, ]+)\*\*", r.answer))
            cited = {c.strip() for x in cited for c in x.split(",")}
            issues = quality_issues(msg, r.answer)
            good = bool(cited & expected) and not issues
            total += 1; passed += good
            print(f"\n> {msg}\n  understood as: {r.standalone}\n  queries: {r.queries}\n"
                  f"  cited {sorted(cited)} (expected any of {sorted(expected)}) "
                  f"{'OK' if good else 'CHECK'}  calls={r.api_calls}")
            if issues:
                print(f"  QUALITY: {'; '.join(issues)}")
            if r.reason:
                print(f"  reason: {r.reason}")
            if r.debug and (not good or r.reason):
                print(f"  raw model output: {r.debug[:400]!r}")
            print("  " + r.answer.replace("\n", "\n  ")[:900])
            for w in r.warnings:
                print("  ! " + w)
            history += [{"role": "user", "content": msg, "standalone": r.standalone, "img_desc": r.image_description},
                        {"role": "assistant", "content": r.answer, "summary": r.summary}]
    print(f"\nLive: {passed}/{total} answers cited an expected page and passed quality checks. "
          f"Total spend: ₹{client.spent:.3f} over {len(client.ledger)} calls")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python eval.py <manual.pdf> [--live] [--only <conversation name part>]")
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    chunks, _ = load_pdf(open(sys.argv[1], "rb").read())
    idx = Index(chunks)
    live(idx, only) if "--live" in sys.argv else sys.exit(0 if offline(idx, only) else 1)
