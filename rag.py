"""Manual ingestion + lexical retrieval.

Why BM25 and not embeddings? Manuals are terse and term-heavy ("white smoke", "clutch
free play", "ABS MIL"); exact-term matching works very well, it's fully local, and it keeps
the whole stack on Sarvam APIs (Sarvam has no embeddings endpoint). The LLM rewrites user
questions (any language, casual wording, image description) into manual-style keywords
before retrieval, which covers most of the vocabulary gap embeddings would normally close.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf as fitz
from rank_bm25 import BM25Okapi

STOP = set("""a an the and or of to in on for with at by from is are was were be been it this that
these those as if then than so do does did can could should would will may might your you
my i me we our us not no yes what why how when where which who whom there here into out up
down over under about please bike motorcycle vehicle""".split())


# Local (zero-token) query expansion: casual owner phrasing -> manual vocabulary.
SYNONYMS = {
    "won't start": "engine does not start starting difficulty starter crank",
    "wont start": "engine does not start starting difficulty starter crank",
    "not starting": "engine does not start starting difficulty starter crank",
    "smoke": "exhaust smoke muffler silencer",
    "overheat": "overheating engine temperature coolant cooling",
    "hot": "overheating temperature",
    "noise": "abnormal noise sound",
    "sound": "noise",
    "vibrat": "vibration",
    "brake": "brake pad fluid lever",
    "stop": "brake",
    "battery": "battery charge terminal",
    "dead": "battery discharged",
    "light": "indicator lamp warning telltale",
    "warning": "indicator lamp warning telltale",
    "mileage": "fuel consumption economy",
    "pick up": "acceleration power",
    "pickup": "acceleration power",
    "leak": "leakage oil seal",
    "chain": "drive chain slack lubrication",
    "clutch": "clutch lever free play",
    "gear": "gear shift transmission",
    "tyre": "tyre tire pressure",
    "tire": "tyre tire pressure",
    "puncture": "tyre tire",
    "headlight": "headlamp bulb",
    "horn": "horn switch fuse",
    "fuse": "fuse blown",
}


def expansion_terms(q: str) -> str:
    ql = q.lower()
    return " ".join(v for k, v in SYNONYMS.items() if k in ql)


def expand_query(q: str) -> str:
    extra = expansion_terms(q)
    return q + (" " + extra if extra else "")


def stem(t: str) -> str:
    """Consistent crude stem: brakes/brake/braking -> brak, smoke/smoking -> smok, leaks -> leak."""
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if len(t) > 5 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        out.append(stem(t))
    return out


@dataclass
class Chunk:
    id: int
    page: int          # PDF page index (1-based)
    heading: str       # section heading, e.g. "CLUTCH CABLE ADJUSTMENT"
    text: str
    label: str = ""    # page number PRINTED in the manual (what the owner sees), falls back to PDF page

    def __post_init__(self):
        self.label = self.label or str(self.page)


INLINE_LABELS = {"CAUTION", "WARNING", "NOTE", "DANGER", "IMPORTANT", "TIP"}


def _is_heading(line: str) -> bool:
    s = line.strip()
    letters = sum(ch.isalpha() for ch in s)
    return (4 <= len(s) <= 70 and s.isupper() and letters >= 0.6 * len(s.replace(" ", ""))
            and not s.endswith(".") and s.rstrip("/: ") not in INLINE_LABELS)


def load_pdf(data: bytes, max_chars: int = 900) -> tuple[list[Chunk], dict]:
    """Section-aware chunking: split on the manual's own ALL-CAPS headings so an excerpt never mixes
    two topics (e.g. a rear-wheel caution leaking into the clutch answer). Running headers/footers
    are removed and the printed page number is kept for citations."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        lines = [re.sub(r"[ \t]+", " ", l).strip() for l in page.get_text("text").splitlines()]
        pages.append([l for l in lines if l])

    # lines repeated on many pages = running header/footer (e.g. "Royal Enfield Classic 350")
    from collections import Counter
    freq = Counter(l for pl in pages for l in set(pl))
    running = {l for l, n in freq.items() if n >= max(3, 0.25 * len(pages)) and not l.isdigit()}

    chunks: list[Chunk] = []
    empty = 0
    heading = ""

    def flush(pno, label, body):
        text = "\n".join(body).strip()
        if len(text) < 25:
            return
        # split long sections on line boundaries
        buf = ""
        for line in text.splitlines():
            if len(buf) + len(line) > max_chars and buf:
                chunks.append(Chunk(len(chunks), pno, heading, buf.strip(), label))
                buf = ""
            buf += line + "\n"
        if buf.strip():
            chunks.append(Chunk(len(chunks), pno, heading, buf.strip(), label))

    for pno, lines in enumerate(pages, start=1):
        label = ""
        for l in lines[:4]:                      # printed page number near the top
            if l.isdigit() and len(l) <= 3:
                label = l
                break
        lines = [l for l in lines if l not in running and l != label]
        if sum(len(l) for l in lines) < 40:
            empty += 1
            continue
        body: list[str] = []
        pending_heading = False
        for l in lines:
            if _is_heading(l):
                if body and not pending_heading:
                    flush(pno, label, body)
                    body = []
                # consecutive heading lines (wrapped or sub-headings) are merged
                heading = (heading + " " + l) if pending_heading else l
                pending_heading = True
            else:
                pending_heading = False
                body.append(l)
        flush(pno, label, body)   # section continues on next page under the same heading
    return chunks, {"pages": len(doc), "chunks": len(chunks), "image_only_pages": empty}


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(c.heading + " " + c.heading + " " + c.text) for c in chunks]  # heading weighted x2
        self.bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, k: int = 6, expansion: str = "", w_exp: float = 0.35):
        """Score = BM25(user terms) + w_exp * BM25(expansion terms) + phrase bonus.
        Keeps synonyms from drowning out what the user actually said."""
        if not self.bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = list(self.bm25.get_scores(q))
        ex = [t for t in tokenize(expansion) if t not in q]
        if ex:
            es = self.bm25.get_scores(ex)
            scores = [a + w_exp * b for a, b in zip(scores, es)]
        # exact adjacent-word phrases from the query (e.g. "white smoke") get a bonus
        words = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in STOP]
        phrases = {f"{a} {b}" for a, b in zip(words, words[1:])}
        if phrases:
            for i, c in enumerate(self.chunks):
                low = re.sub(r"\s+", " ", c.text.lower())
                hits = sum(1 for ph in phrases if ph in low)
                if hits:
                    scores[i] += 2.0 * hits
        # the manual's troubleshooting table is the most authoritative place for symptoms
        for i, c in enumerate(self.chunks):
            if scores[i] > 0 and "TROUBLESHOOT" in c.heading:
                scores[i] *= 1.6
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
