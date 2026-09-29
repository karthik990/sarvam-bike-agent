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
        ("is it the same with a pillion?", ["tyre pressure with pillion"], {"71"}),
        ("And the engine oil grade?", ["engine oil grade"], {"19"}),
        ("how often should I change it?", ["engine oil replace periodical maintenance"], {"111"}),
        ("How much chain slack is allowed?", ["drive chain slackness"], {"84"}),
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


def offline(index):
    total = ok = 0
    for name, turns in CONVERSATIONS.items():
        print(f"\n=== {name}")
        for msg, queries, expected in turns:
            sections, _, _ = agent.retrieve(index, queries, None)
            got = {l for s in sections for l in s["labels"]}
            hit = bool(expected & got)
            total += 1; ok += hit
            print(f"  {'PASS' if hit else 'FAIL'}  {msg[:55]:55} want {sorted(expected)} got {sorted(got, key=lambda x: int(x) if x.isdigit() else 0)}")
    print(f"\nRetrieval: {ok}/{total} questions reached the right page(s)")
    return ok == total


def live(index):
    from sarvam_client import Sarvam
    load_dotenv()
    client = Sarvam(os.environ["SARVAM_API_KEY"])
    for name, turns in CONVERSATIONS.items():
        print(f"\n=== {name}")
        history = []
        for msg, _, expected in turns:
            r = agent.answer(client, index, msg, history)
            cited = set(re.findall(r"\(p\.(\d+)\)", r.answer))
            print(f"\n> {msg}\n  understood as: {r.standalone}\n  queries: {r.queries}\n"
                  f"  cited {sorted(cited)} (expected any of {sorted(expected)}) "
                  f"{'OK' if cited & expected else 'CHECK'}  calls={r.api_calls}")
            print("  " + r.answer.replace("\n", "\n  ")[:900])
            for w in r.warnings:
                print("  ! " + w)
            history += [{"role": "user", "content": msg, "standalone": r.standalone, "img_desc": r.image_description},
                        {"role": "assistant", "content": r.answer, "summary": r.summary}]
    print(f"\nTotal spend: ₹{client.spent:.3f} over {len(client.ledger)} calls")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python eval.py <manual.pdf> [--live]")
    chunks, _ = load_pdf(open(sys.argv[1], "rb").read())
    idx = Index(chunks)
    live(idx) if "--live" in sys.argv else sys.exit(0 if offline(idx) else 1)
