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


# HELD-OUT set: questions NOT used while tuning. Casual owner wording; facts taken from the PDF text.
# (message, expected pages or None = must refuse / say not covered, facts that must appear)
HOLDOUT = {
    "specs (casual wording)": [
        ("How much petrol does the tank hold?", {"16"}, ["13"]),
        ("At what level does the low fuel warning come on?", {"16"}, ["4"]),
        ("How heavy is the bike?", {"18"}, ["195"]),
        ("What's the ground clearance?", {"18"}, ["170"]),
        ("What battery does it have?", {"84", "17"}, ["8"]),
        ("What wattage is the halogen headlamp?", {"17"}, ["60/55"]),
        ("What should the spark plug gap be?", {"14"}, ["0.7"]),
    ],
    "maintenance how-tos": [
        ("Which brake fluid should I use?", {"19", "69"}, ["DOT 4"]),
        ("How do I check the brake fluid level?", {"68", "69"}, ["MAX"]),
        ("Where is the air filter and how do I get to it?", {"101"}, ["left side panel"]),
        ("Any precautions when washing the bike?", {"105", "106"}, ["cold"]),
        ("I won't ride for two months, how should I store it?", {"107"}, ["month"]),
    ],
    "new bike / electronics": [
        ("Anything special I should do in the first few hundred km?", {"53", "54"}, []),
        ("The engine warning light stays on after starting", {"54", "55", "109"}, ["service"]),
        ("How do I connect my phone to the navigation pod?", {"41", "42", "43", "44", "45"}, ["bluetooth"]),
    ],
    "follow-up chain: fork oil": [
        ("What oil goes in the front forks?", {"19"}, []),
        ("and how often should it be replaced?", {"113"}, ["20", "40"]),
    ],
    "follow-up chain: spark plug": [
        ("What's the recommended spark plug?", {"14"}, ["YR7MES"]),
        ("when does it need replacing?", {"112"}, ["20"]),
    ],
    "hindi + multi-question": [
        ("टायर प्रेशर कितना होना चाहिए?", {"71", "16", "104"}, ["32"]),
        ("What's the fuel tank capacity and the kerb weight?", {"16", "18"}, ["13", "195"]),
    ],
    "must refuse (not in the manual)": [
        ("How do I fix a puncture on the road?", None, []),
        ("What is the top speed of this bike?", None, []),
        ("What mileage does it give per litre?", None, []),
    ],
}


# Held-out set for a DIFFERENT manufacturer: Honda Shine 100 owner's manual (Apr 2023).
# Written without looking at retrieval results; facts/pages taken from the PDF text (printed page numbers).
HONDA_SHINE100 = {
    "specs (casual wording)": [
        ("What tyre pressure should I run?", {"100"}, ["25", "33"]),
        ("What's the tyre pressure when I carry a passenger?", {"100"}, ["41"]),
        ("How many litres does the tank take?", {"99"}, ["9"]),
        ("How much does the bike weigh?", {"99"}, ["99"]),
        ("What's the ground clearance?", {"99"}, ["168"]),
        ("Which spark plug does it use and what gap?", {"100", "50"}, ["0.8"]),
        ("Which engine oil should I use?", {"100", "38"}, ["10W-30"]),
        ("How much oil does the engine take?", {"100"}, ["0.75"]),
        ("How much drive chain slack is allowed?", {"100", "64", "65", "66", "67"}, ["20"]),
        ("What rating is the main fuse?", {"101", "85"}, ["15"]),
    ],
    "maintenance intervals (schedule table)": [
        ("How often should the engine oil be changed?", {"32"}, ["6"]),
        ("When should I replace the spark plug?", {"32"}, ["12"]),
        ("How often do I replace the air filter?", {"32"}, ["18"]),
    ],
    "problems": [
        ("My bike won't start", {"74"}, []),
        ("I got a puncture, what should I do?", {"76", "77", "78", "79", "80", "81"}, []),
        ("The battery keeps going dead", {"82"}, []),
        ("The engine warning light is blinking", {"75"}, []),
    ],
    "follow-up chain: oil": [
        ("What oil should I use?", {"100", "38"}, ["10W-30"]),
        ("and how much of it goes in?", {"100"}, ["0.75"]),
    ],
    "follow-up chain: chain": [
        ("How do I adjust the drive chain?", {"64", "65", "66", "67"}, []),
        ("how often should it be lubricated?", {"32", "64", "65", "66", "67"}, ["500"]),
    ],
    "real user conversation (topic switches, short replies, corrections)": [
        ("bike doesn't start", {"74"}, []),
        ("no petrol", {"31", "11", "12"}, []),
        ("how to put petrol", {"31"}, []),
        ("lights don't work", {"82", "83", "84", "85", "77"}, []),
        ("no my bike lights don't work", {"82", "83", "84", "85", "77"}, []),
    ],
    "hindi + multi-question": [
        ("टायर प्रेशर कितना होना चाहिए?", {"100"}, ["25"]),
        ("What's the tank capacity and the kerb weight?", {"99"}, ["9", "99"]),
    ],
    "must refuse (not in this manual)": [
        ("What is the top speed?", None, []),
        ("How do I pair my phone over Bluetooth?", None, []),
        ("What mileage does it give per litre?", None, []),
    ],
}
SETS = {"re": HOLDOUT, "honda": HONDA_SHINE100}


def holdout_offline(index, dataset=None):
    dataset = dataset or HOLDOUT
    """Retrieval-only check on FIRST turns with the raw message (no planner involved)."""
    total = ok = 0
    for name, turns in dataset.items():
        print(f"\n=== {name}")
        for i, (msg, expected, _) in enumerate(turns):
            if agent.detect_lang(msg) != "en-IN":
                print(f"  ----  {msg[:55]:55} (non-English: translated by the planner, checked in --live)")
                continue
            if i > 0 and name.startswith("follow-up"):
                print(f"  ----  {msg[:55]:55} (follow-up: needs the planner, checked in --live)")
                continue
            sections, _, per_query = agent.retrieve(index, [msg], None)
            got = {l for s in sections for l in s["labels"]}
            best = max([sc for _, _, hs in per_query for _, sc in hs], default=0)
            if expected is None:
                print(f"  INFO  {msg[:55]:55} best score {best:.1f} (model must decline) pages {sorted(got, key=lambda x: int(x) if x.isdigit() else 0)}")
                continue
            hit = bool(expected & got)
            total += 1; ok += hit
            print(f"  {'PASS' if hit else 'FAIL'}  {msg[:55]:55} want {sorted(expected)} got {sorted(got, key=lambda x: int(x) if x.isdigit() else 0)}")
    print(f"\nHeld-out retrieval (raw first questions): {ok}/{total}")
    return ok == total


def holdout_live(index, dataset=None):
    dataset = dataset or HOLDOUT
    from sarvam_client import Sarvam
    load_dotenv()
    client = Sarvam(os.environ["SARVAM_API_KEY"])
    passed = total = 0
    for name, turns in dataset.items():
        print(f"\n=== {name}")
        history = []
        for msg, expected, must in turns:
            r = agent.answer(client, index, msg, history)
            cited = agent.cited_pages(r.answer) | {c.strip() for x in re.findall(r"\*\*p\.([\d, ]+)\*\*", r.answer)
                                                    for c in x.split(",")}
            low = r.answer.lower()
            if expected is None:
                declined = (not r.found) or ("not covered" in low and not cited)
                good, verdict = declined, ("declined correctly" if declined else "SHOULD HAVE DECLINED")
            else:
                missing = [m for m in must if m.lower() not in low]
                missing += [f"contains '{f}'" for f in QUALITY.get(msg, {}).get("forbid", []) if f.lower() in low]
                good = bool(cited & expected) and not missing
                verdict = "OK" if good else f"CHECK (cited {sorted(cited)}, want {sorted(expected)}" + \
                          (f", missing {missing})" if missing else ")")
            total += 1; passed += good
            print(f"\n> {msg}\n  understood as: {r.standalone} | queries: {r.queries} | calls={r.api_calls}\n  {verdict}")
            if r.reason:
                print(f"  reason: {r.reason}")
            if not good and r.debug:
                print(f"  raw model output: {r.debug[:300]!r}")
            print("  " + r.answer.replace("\n", "\n  ")[:700])
            history += [{"role": "user", "content": msg, "standalone": r.standalone, "img_desc": r.image_description},
                        {"role": "assistant", "content": r.answer, "summary": r.summary}]
    print(f"\nHeld-out live: {passed}/{total} correct. Spend: ₹{client.spent:.3f} over {len(client.ledger)} calls")


# Answer-quality checks for the live run (beyond "cited the right page")
QUALITY = {
    "lights don't work": {"forbid": ["tank capacity", "fuel fill cap", "9.0 litres"]},
    "no my bike lights don't work": {"forbid": ["tank capacity", "fuel fill cap", "9.0 litres"]},
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
    bad = agent.cited_pages(answer) & q.get("avoid_pages", set())
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
            cited = agent.cited_pages(r.answer) | {c.strip() for x in re.findall(r"\*\*p\.([\d, ]+)\*\*", r.answer)
                                                    for c in x.split(",")}
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
        sys.exit("usage: python eval.py <manual.pdf> [--live] [--holdout | --set re|honda] [--only <conversation name part>]")
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    chunks, _ = load_pdf(open(sys.argv[1], "rb").read())
    idx = Index(chunks)
    if "--holdout" in sys.argv or "--set" in sys.argv:
        ds = SETS[sys.argv[sys.argv.index("--set") + 1]] if "--set" in sys.argv else HOLDOUT
        holdout_live(idx, ds) if "--live" in sys.argv else sys.exit(0 if holdout_offline(idx, ds) else 1)
    else:
        live(idx, only) if "--live" in sys.argv else sys.exit(0 if offline(idx, only) else 1)
