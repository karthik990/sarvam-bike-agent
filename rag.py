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
    "how often": "maintenance schedule periodic periodical service interval km months",
    "interval": "maintenance schedule periodic periodical service km months",
    "when should": "maintenance schedule periodic periodical service interval km months",
    "service schedule": "maintenance schedule periodic periodical km months",
    "change it": "replace",
    "does not start": "engine does not start",
    "petrol": "fuel",
    "gas ": "fuel",
    "tank hold": "fuel tank capacity",
    "hold": "capacity",
    "heavy": "weight kerb weight",
    "weigh": "weight kerb weight",
    "how tall": "height seat height",
    "phone": "bluetooth mobile app usb charging",
    "navigation": "navigation bluetooth app",
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


TS_RE = re.compile(r"trouble\s*-?\s*shoot|fault\s*finding|diagnos", re.I)   # troubleshooting section, any brand
SCHEDULE_RE = re.compile(r"maintenance item|maintenance schedule|periodic(al)? maintenance|service schedule|"
                         r"maintenance chart|service chart|maintenance interval", re.I)


def _is_heading(line: str, size: float | None = None, bold: bool = False, body: float | None = None,
                modes: frozenset = frozenset({"caps"})) -> bool:
    """A heading in the style THIS manual uses (see choose_heading_modes): ALL CAPS, noticeably larger
    font, or short bold title-like lines ('Tyre Pressure', 'Checking Engine Oil')."""
    s = line.strip()
    if not (3 <= len(s) <= 80) or s.endswith((".", ",", ";")) or re.fullmatch(r"[\d\W]+", s):
        return False
    if s.rstrip("/: ").upper() in INLINE_LABELS:
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.6 * len(s.replace(" ", "")):
        return False
    words = s.split()
    caps = "caps" in modes and len(s) >= 4 and s.isupper()
    big = "big" in modes and bool(size and body and size >= body * 1.15 and len(words) <= 12)
    bold_title = "bold" in modes and bool(bold and len(words) <= 10 and s[0].isupper() and not s.endswith(":"))
    return caps or big or bold_title


def choose_heading_modes(styled_pages, body: float | None) -> frozenset:
    """Pick the heading signal this particular manual uses consistently:
    many ALL-CAPS title lines -> caps (keeps table headers / bold body text from splitting sections);
    otherwise larger-font lines; bold short titles only if font size doesn't separate headings."""
    n_pages = max(1, len(styled_pages))
    caps = big = bold = 0
    for sl in styled_pages:
        for t, sz, bd in sl:
            if _is_heading(t, sz, bd, body, frozenset({"caps"})):
                caps += 1
            if _is_heading(t, sz, bd, body, frozenset({"big"})):
                big += 1
            if _is_heading(t, sz, bd, body, frozenset({"bold"})):
                bold += 1
    if caps >= 0.5 * n_pages:
        return frozenset({"caps"})
    if big >= 0.3 * n_pages:
        return frozenset({"big", "caps"})
    return frozenset({"big", "caps", "bold"})


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


_NUM = re.compile(r"^\s*(\d{1,3}(?:[,\s]\d{3})*|\d+(?:\.\d+)?)\s*(k|km|kms|thousand)?\s*$", re.I)


def _is_num(cell: str) -> bool:
    return bool(_NUM.match(cell or ""))


def generic_schedule(page, page_text: str) -> list[str]:
    """Rebuild ANY service/maintenance schedule table (not just one brand's layout) into sentences.
    Finds a header row of odometer values (e.g. 0.5/5/10 or 750/3,000/6,000), an optional months row,
    and rows of short codes (R/I/C/A/L, check marks...). Codes are expanded with the legend printed
    on the page when there is one ('I = Inspect', 'R : Replace')."""
    import contextlib, io
    if not re.search(r"\bkm|kms|kilomet|odometer", page_text, re.I):
        return []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tables = page.find_tables().tables
    except Exception:
        return []
    legend = {}
    for code, word in re.findall(r"(?<![A-Za-z])([A-Z]{1,2}|[✓✔•*])\s*[:=–-]\s*([A-Z][a-z]+(?:\s+[a-z]+)?)", page_text):
        legend.setdefault(code, word)
    if len(legend) < 2:
        legend = dict(CODE_NAMES) if re.search(r"\binspect", page_text, re.I) else {}
    out = []
    for t in tables:
        rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
        header_i = next((i for i, r in enumerate(rows) if sum(_is_num(c) for c in r) >= 3), None)
        if header_i is None:
            continue
        cols = [j for j, c in enumerate(rows[header_i]) if _is_num(c)]
        head_txt = " ".join(rows[header_i]).lower() + " " + page_text.lower()[:2000]
        unit = " thousand km" if re.search(r"x\s*1[,.]?000", head_txt) else (" km" if "km" in head_txt else "")
        months = None
        for r in rows[header_i + 1:header_i + 3]:
            if any("month" in c.lower() for c in r) and sum(_is_num(r[j]) for j in cols if j < len(r)) >= 3:
                months = [r[j] if j < len(r) else "" for j in cols]
        for r in rows[header_i + 1:]:
            if months is not None and any("month" in c.lower() for c in r):
                continue
            codes = [r[j] if j < len(r) else "" for j in cols]
            if not any(codes) or any(len(c) > 8 for c in codes):
                continue
            name_cells = [c for j, c in enumerate(r) if j not in cols and c and not c.isdigit()]
            if not name_cells:
                continue
            name = max(name_cells, key=len)
            groups: dict[str, list[tuple[str, str]]] = {}
            for j, c in enumerate(codes):
                if c:
                    groups.setdefault(c, []).append((rows[header_i][cols[j]], months[j] if months else ""))
            parts = []
            for c, vals in groups.items():
                word = " & ".join(legend.get(x.strip(), x.strip()) for x in re.split(r"&|/", c))
                at = ", ".join(v for v, _ in vals)
                mo = ", ".join(m for _, m in vals if m)
                parts.append(f"{word} at {at}{unit}" + (f" ({mo} months)" if mo else ""))
            out.append(f"Maintenance item: {name}: " + "; ".join(parts) + ".")
            generic_schedule.used_labels.update({name} | {c for c in rows[header_i] if c and not _is_num(c)})
            if months is not None:
                generic_schedule.used_labels.update(c for c in rows[header_i + 1] if c and not _is_num(c))
    return out


generic_schedule.used_labels = set()


def _page_label(lines: list[str]) -> str:
    """Printed page number: checked at the top AND bottom of the page ('12', 'Page 12', '- 12 -')."""
    for l in lines[:4] + lines[-4:][::-1]:
        m = re.fullmatch(r"(?:page\s*)?[-–]?\s*(\d{1,3})\s*[-–]?", l.strip(), re.I)
        if m:
            return m.group(1)
    return ""


def _styled_lines(page) -> list[tuple[str, float, bool]]:
    """(text, font size, bold) per visual line."""
    out = []
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            spans = [sp for sp in ln.get("spans", []) if sp.get("text", "").strip()]
            if not spans:
                continue
            text = re.sub(r"[ \t]+", " ", "".join(sp["text"] for sp in ln["spans"])).strip()
            size = max(sp.get("size", 0) for sp in spans)
            bold = any((sp.get("flags", 0) & 16) or "bold" in sp.get("font", "").lower() for sp in spans)
            out.append((text, size, bold))
    return out


def load_pdf(data: bytes, max_chars: int = 900) -> tuple[list[Chunk], dict]:
    """Section-aware chunking for ANY owner's/service manual: split on the manual's own headings
    (detected from layout: ALL CAPS, larger font or short bold titles) so an excerpt never mixes two
    topics. Running headers/footers are removed, the printed page number (top or bottom) is kept for
    citations, and maintenance-schedule tables are rebuilt into sentences."""
    doc = fitz.open(stream=data, filetype="pdf")
    styled = [_styled_lines(page) for page in doc]
    pages = [[t for t, _, _ in sl] for sl in styled]
    style_of = [{t: (sz, bd) for t, sz, bd in sl} for sl in styled]
    from collections import Counter
    sizes = Counter()
    for sl in styled:
        for t, sz, _ in sl:
            sizes[round(sz, 1)] += len(t)
    body_size = sizes.most_common(1)[0][0] if sizes else None
    modes = choose_heading_modes(styled, body_size)

    # lines repeated on many pages = running header/footer (model name, chapter title...)
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
        label = _page_label(lines)
        lines = [l for l in lines if l not in running and l != label]
        if "km (x" in " ".join(lines).lower() or mstate.get("km"):
            mstate["grid"] = maintenance_grid(doc[pno - 1])
        lines = normalize_maintenance(lines, mstate)
        if not mstate.get("km") and not any(l.startswith("Maintenance item") for l in lines):
            generic_schedule.used_labels = set()
            sched = generic_schedule(doc[pno - 1], "\n".join(lines))
            if sched:
                # table cells (codes, odometer numbers, row labels) are replaced by the rebuilt sentences
                used = generic_schedule.used_labels
                lines = [l for l in lines if l not in used and
                         not (len(l) <= 8 and (re.fullmatch(r"[A-Z&/✓✔•*]{1,5}", l) or _is_num(l)))]
                lines += sched
        if sum(len(l) for l in lines) < 40:
            empty += 1
            continue
        body: list[str] = []
        pending_heading = False
        st = style_of[pno - 1]
        for l in lines:
            sz, bd = st.get(l, (None, False))
            if _is_heading(l, sz, bd, body_size, modes):
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
    return chunks, {"pages": len(doc), "chunks": len(chunks), "image_only_pages": empty,
                    "heading_style": "+".join(sorted(modes))}


SYNONYM_ONLY_MIN = 2.5


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
        own = [sc > 0 for sc in scores]          # does the chunk contain any of the OWNER'S words?
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
                if scores[i] > 0 and TS_RE.search(c.heading):
                    scores[i] *= 1.6
        # synonyms boost a section; on their own they count only when the match is strong
        # ('how heavy' -> 'kerb weight' yes; 'phone' -> 'charging' -> 'charge the battery' no)
        scores = [sc if (own[i] or sc >= SYNONYM_ONLY_MIN) else 0.0 for i, sc in enumerate(scores)]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


# stems of words that describe a SYMPTOM (used to decide when to favour the troubleshooting table)
SYMPTOM_TERMS = {stem(w) for w in """start starting stop stops shut shuts misfire misfires erratic pickup
abs mil lamp light lights dim horn fuse blown battery hot overheat overheating smoke noise vibration""".split()}
