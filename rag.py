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
    "how often": "periodical maintenance interval km months schedule",
    "interval": "periodical maintenance km months schedule",
    "when should": "periodical maintenance interval km months",
    "service schedule": "periodical maintenance km months",
    "change it": "replace",
    "does not start": "engine does not start",
    "petrol": "fuel",
    "gas ": "fuel",
    "tank hold": "fuel tank capacity",
    "hold": "capacity",
    "heavy": "weight kerb weight",
    "weigh": "weight kerb weight",
    "how tall": "height seat height",
    "phone": "bluetooth mobile app tripper",
    "navigation": "tripper bluetooth",
    "engine light": "MIL malfunction indicator lamp",
    "warning light": "indicator lamp MIL",
    "wash": "washing cleaning",
    "store": "storage",
    "first few": "running in period",
    "new bike": "running in period",
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


CODE_NAMES = {"R": "Replace", "I": "Inspect", "C": "Clean", "A": "Adjust", "L": "Lubricate"}
CODE_RE = re.compile(r"^(?:[RICAL])(?:\s*&\s*[RICAL])?$")


def _fmt_code(c: str) -> str:
    return " & ".join(CODE_NAMES.get(x.strip(), x.strip()) for x in c.split("&"))


def maintenance_grid(page) -> dict[int, list[str]]:
    """Exact cell grid of a periodic-maintenance chart (blank cells kept) via PyMuPDF find_tables.
    Returns {row_number: [code per service column]}; {} if the page has no such table."""
    import contextlib, io
    grid = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tables = page.find_tables().tables
    except Exception:
        return grid
    for t in tables:
        rows = t.extract()
        if not any((r[1] or "").lower().startswith("km (x") for r in rows if len(r) > 2):
            continue
        for r in rows:
            if r and (r[0] or "").strip().isdigit() and len(r) > 2:
                cells = [(c or "").strip().replace(" ", "").replace("l", "I") for c in r[2:]]
                if all(c == "" or CODE_RE.match(c) for c in cells):
                    grid[int(r[0])] = cells
    return grid


def normalize_maintenance(lines: list[str], state: dict) -> list[str]:
    """Periodic-maintenance charts lose their column alignment when a PDF is flattened
    ('Engine oil R I R I R ...'). Rebuild each row as a plain sentence using the chart's km/month
    header, e.g. 'Engine oil: Replace at 0.5, 10, 20 ... thousand km (1.5, 12, 24 ... months);
    Inspect at 5, 15, ...'. Rows whose cells can't be aligned are flagged instead of guessed."""
    text = " ".join(lines).lower()
    if "km (x 1,000)" in text or "km (x1,000)" in text:
        i = next(k for k, l in enumerate(lines) if l.lower().startswith("km (x"))
        km, months, j = [], [], i + 1
        while j < len(lines) and re.fullmatch(r"\d+(\.\d+)?", lines[j]):
            km.append(lines[j]); j += 1
        if j < len(lines) and lines[j].lower().startswith("month"):
            j += 1
            while j < len(lines) and len(months) < len(km) and re.fullmatch(r"\d+(\.\d+)?", lines[j]):
                months.append(lines[j]); j += 1
        if km and len(months) == len(km):
            state.update(km=km, months=months)
            head = lines[:i] + [f"Service columns: {', '.join(km)} thousand km / {', '.join(months)} months (whichever is earlier)."]
            lines = head + lines[j:]
        else:
            return lines
    if not state.get("km"):
        return lines
    km, months = state["km"], state["months"]
    lines = ["I" if l == "l" else l for l in lines]      # PDF renders one 'I' cell as lowercase 'l'
    out, k, expected = [], 0, None
    while k < len(lines):
        l = lines[k]
        is_row = re.fullmatch(r"\d{1,2}", l) and (expected is None or int(l) == expected)
        if not is_row:
            out.append(l); k += 1; continue
        num = int(l); expected = num + 1; k += 1
        name, codes, notes = [], [], []
        while k < len(lines) and not CODE_RE.match(lines[k]) and not re.fullmatch(r"\d{1,2}", lines[k]):
            name.append(lines[k]); k += 1
        while k < len(lines) and CODE_RE.match(lines[k]):
            codes.append(lines[k].replace(" ", "")); k += 1
        while k < len(lines) and not (re.fullmatch(r"\d{1,2}", lines[k]) and int(lines[k]) == expected):
            notes.append(lines[k]); k += 1
        nm = " ".join(name).strip()
        g = state.get("grid", {}).get(num)
        if g and len(g) == len(km) and [c for c in g if c] == codes:
            # exact grid from the table reader: skip blank cells, keep true column positions
            pairs = [(c, a, b) for c, a, b in zip(g, km, months) if c]
            groups = {}
            for c, a, b in pairs:
                groups.setdefault(c, ([], []))
                groups[c][0].append(a); groups[c][1].append(b)
            parts = [f"{_fmt_code(c)} at {', '.join(a)} thousand km ({', '.join(b)} months)" for c, (a, b) in groups.items()]
            row = f"Maintenance item {num}. {nm}: " + "; ".join(parts) + "."
        elif len(codes) == len(km):
            groups = {}
            for c, a, b in zip(codes, km, months):
                groups.setdefault(c, ([], []))
                groups[c][0].append(a); groups[c][1].append(b)
            parts = [f"{_fmt_code(c)} at {', '.join(a)} thousand km ({', '.join(b)} months)" for c, (a, b) in groups.items()]
            row = f"Maintenance item {num}. {nm}: " + "; ".join(parts) + "."
        elif codes:
            row = (f"Maintenance item {num}. {nm}: chart marks {', '.join(_fmt_code(c) for c in codes)} in "
                   f"{len(codes)} of the {len(km)} service columns (exact km per the printed chart).")
        else:
            row = f"Maintenance item {num}. {nm}"
        if notes:
            row += " " + " ".join(notes)
        out.append(row)
    return out


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
    mstate: dict = {}

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
        if "km (x" in " ".join(lines).lower() or mstate.get("km"):
            mstate["grid"] = maintenance_grid(doc[pno - 1])
        lines = normalize_maintenance(lines, mstate)
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
        self._flat = [re.sub(r"[^a-z0-9]+", " ", (c.heading + " " + c.text).lower()) for c in chunks]

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
        # exact phrases get a bonus. Built from the RAW words (stop-words kept) so "engine does not
        # start" matches that troubleshooting row rather than "engine starts but shuts off".
        def grams(text, weight3, weight2):
            raw = re.findall(r"[a-z0-9]+", text.lower())
            out = {}
            for i in range(len(raw) - 2):              # trigrams with at least one content word
                g = raw[i:i + 3]
                if any(w not in STOP for w in g):
                    out[" ".join(g)] = weight3
            for i in range(len(raw) - 1):              # bigrams: two content words, or a negation
                a, b = raw[i], raw[i + 1]
                if (a not in STOP and b not in STOP) or (a in ("not", "no") and b not in STOP):
                    out.setdefault(f"{a} {b}", weight2)
            return out

        phrases = grams(query, 3.0, 2.0)
        for ph, w in grams(expansion, 0.0, 1.0).items():   # manual vocabulary from expansion, weaker
            phrases.setdefault(ph, w)
        phrases = {k: v for k, v in phrases.items() if v > 0}
        if phrases:
            for i in range(len(self.chunks)):
                low = self._flat[i]
                bonus = sum(w for ph, w in phrases.items() if ph in low)
                if bonus:
                    scores[i] += bonus
        # the troubleshooting table is authoritative for SYMPTOMS (not for maintenance questions)
        if set(q) & SYMPTOM_TERMS:
            for i, c in enumerate(self.chunks):
                if scores[i] > 0 and "TROUBLESHOOT" in c.heading:
                    scores[i] *= 1.6
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


# stems of words that describe a SYMPTOM (used to decide when to favour the troubleshooting table)
SYMPTOM_TERMS = {stem(w) for w in """start starting stop stops shut shuts misfire misfires erratic pickup
abs mil lamp light lights dim horn fuse blown battery hot overheat overheating smoke noise vibration""".split()}
