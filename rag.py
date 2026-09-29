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
    "heavy": "weight kerb curb weight",
    "weigh": "weight kerb curb weight",
    "take": "capacity",
    "litres": "capacity",
    "liters": "capacity",
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
    # part numbers / codes in capitals ('CPR7EA-9 (NGK)', 'ETZ4 / ATZ4L') are table values, not headings
    codey = sum(bool(re.search(r"[A-Za-z]", w) and re.search(r"\d", w)) for w in words) >= max(1, len(words) / 2)
    emphasised = bool(bold or (size and body and size >= body * 1.05))
    caps = ("caps" in modes and len(s) >= 4 and s.isupper() and not codey
            and (modes == frozenset({"caps"}) or emphasised))    # mixed-style manual: caps must also stand out
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


def _legend(page_text: str) -> dict:
    """Code legend printed on the page: 'I : Inspect', 'R = Replace', 'I = Inspect R = Replace ...'.
    Only ':' or '=' count as separators (a '–' is an empty table cell in many manuals, not a legend)."""
    legend = {}
    for code, word in re.findall(r"(?<![A-Za-z])([A-Z]{1,2}|[✓✔•*])\s*[:=]\s*([A-Z][a-z]+)", page_text):
        legend.setdefault(code, word)
    if len(legend) < 2:
        legend = dict(CODE_NAMES) if re.search(r"\binspect", page_text, re.I) else {}
    return legend


def _cx(cell) -> float:
    return (cell[0] + cell[2]) / 2


def generic_schedule(page, page_text: str, doc_legend: dict | None = None) -> list[str]:
    """Rebuild ANY service/maintenance schedule table into sentences.

    Columns are matched by POSITION ON THE PAGE, not by column index: many manuals split the grid
    finely, so a code printed under '6' can sit in a different grid column than the '6' heading.
    Handles: odometer header rows ('× 1,000 km 1 6 12 ...' or '750 3,000 6,000'), a months row,
    extra header rows (miles), extra labelled columns ('Pre-ride Check', 'Annual Check'), note cells
    ('500 km (300mi): I L') and the legend printed on the page."""
    import contextlib, io
    if not re.search(r"\bkm|kms|kilomet|odometer", page_text, re.I):
        return []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tables = page.find_tables().tables
    except Exception:
        return []
    legend = _legend(page_text)
    if doc_legend and (len(legend) < 2 or legend == CODE_NAMES):
        legend = doc_legend                     # legend printed on another page of the same manual
    word = lambda c: " & ".join(legend.get(x.strip(), x.strip()) for x in re.split(r"&|/", c) if x.strip())
    out = []
    for t in tables:
        texts = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
        boxes = [r.cells for r in t.rows]
        num_rows = [i for i, r in enumerate(texts) if sum(_is_num(c) for c in r) >= 3]
        if not num_rows:
            continue
        h = num_rows[0]
        head = [(texts[h][j], boxes[h][j]) for j in range(len(texts[h])) if _is_num(texts[h][j]) and boxes[h][j]]
        if len(head) < 3:
            continue
        row_label = " ".join(c for c in texts[h] if c and not _is_num(c)).lower()
        unit_src = row_label + " " + page_text.lower()[:3000]
        unit = " thousand km" if re.search(r"[x×]\s*1[,.\s]?000", unit_src) else (" km" if "km" in unit_src else "")
        span_lo, span_hi = min(b[0] for _, b in head), max(b[2] for _, b in head)

        def col_of(cell):
            """header value whose cell contains this cell's centre (else the nearest one)."""
            x = _cx(cell)
            for v, b in head:
                if b[0] - 1 <= x <= b[2] + 1:
                    return v
            v, b = min(head, key=lambda hb: abs(_cx(hb[1]) - x))
            return v if abs(_cx(b) - x) <= (b[2] - b[0]) else None

        months = {}
        for i in num_rows[1:]:
            if any("month" in c.lower() for c in texts[i]):
                for j, c in enumerate(texts[i]):
                    if _is_num(c) and boxes[i][j]:
                        v = col_of(boxes[i][j])
                        if v:
                            months[v] = c
        # extra labelled columns from the rows above the header ('Pre-ride Check', 'Annual Check')
        extra = []
        for i in range(h):
            for j, c in enumerate(texts[i]):
                b = boxes[i][j]
                if c and b and not (span_lo - 1 <= _cx(b) <= span_hi + 1) and not re.match(r"items?$|refer|page", c, re.I):
                    extra.append((re.sub(r"\s+P\.?$", "", c).strip(), b))
        for i, r in enumerate(texts):
            if i <= h or i in num_rows:
                continue
            cells = [(c, boxes[i][j]) for j, c in enumerate(r) if c and boxes[i][j]]
            if not cells:
                continue
            name = cells[0][0]
            if len(name) <= 3 or _is_num(name):
                continue
            inside = [(c, b) for c, b in cells[1:] if span_lo - 1 <= _cx(b) <= span_hi + 1]
            outside = [(c, b) for c, b in cells[1:] if not (span_lo - 1 <= _cx(b) <= span_hi + 1)]
            parts = []
            for c, b in outside:                      # codes under extra labelled columns
                if len(c) <= 3 and not c.isdigit() and c not in "–-":
                    lab = next((l for l, lb in extra if lb[0] - 2 <= _cx(b) <= lb[2] + 2), "")
                    if lab:
                        parts.append(f"{word(c)} ({lab.lower()})")
            if any(len(c) > 3 and not _is_num(c) for c, _ in inside):
                # a note spanning the odometer columns, e.g. '500 km (300mi): I L'
                note = " ".join(c for c, _ in inside)
                note = re.sub(r"(?<![A-Za-z])([A-Z]{1,2})(?![A-Za-z])", lambda m: legend.get(m.group(1), m.group(1)), note)
                parts.append(f"every {note}")
            else:
                groups: dict[str, list[str]] = {}
                for c, b in inside:
                    if c in "–-" or len(c) > 3:
                        continue
                    v = col_of(b)
                    if v and v not in groups.get(c, []):
                        groups.setdefault(c, []).append(v)
                for c, vals in groups.items():
                    mo = [months[v] for v in vals if v in months]
                    parts.append(f"{word(c)} at {', '.join(vals)}{unit}" + (f" ({', '.join(mo)} months)" if mo else ""))
            if parts:
                out.append(f"Maintenance item: {name}: " + "; ".join(parts) + ".")
                generic_schedule.used_labels.update({name} | {c for c, _ in cells})
        for i in range(h + 1):
            generic_schedule.used_labels.update(c for c in texts[i] if c)
        for i in num_rows:
            generic_schedule.used_labels.update(c for c in texts[i] if c and not _is_num(c))
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
    doc_legend = {}
    for pl in pages:                            # a code legend printed anywhere applies to the whole manual
        lg = _legend("\n".join(pl))
        if len(lg) >= 3 and lg != CODE_NAMES:
            doc_legend = lg
            break
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
            sched = generic_schedule(doc[pno - 1], "\n".join(lines), doc_legend)
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
                heading = (heading + " " + l) if pending_heading and l not in heading else (heading if pending_heading else l)
                pending_heading = True
            else:
                pending_heading = False
                body.append(l)
        flush(pno, label, body)   # section continues on next page under the same heading
    return chunks, {"pages": len(doc), "chunks": len(chunks), "image_only_pages": empty,
                    "heading_style": "+".join(sorted(modes))}


SYNONYM_ONLY_MIN = 2.5
SPEC_HEAD_RE = re.compile(r"specification|technical data|service data|spec sheet", re.I)
VALUE_Q_RE = re.compile(r"\b(what|which|how much|how many|how heavy|capacity|pressure|gap|weight|weigh|size|"
                        r"rating|grade|dimension|clearance|litres?|liters?|wattage|voltage)\b", re.I)
ACTION_Q_RE = re.compile(r"\b(how (do|to|can|should) i|adjust|replace|remove|install|fix|won'?t|doesn'?t|not work|"
                         r"what should i do|what to do|how often|when should)\b", re.I)


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
        # the specifications section is authoritative for VALUE questions ('what tyre pressure',
        # 'how many litres', 'which spark plug'); every manual has one, whatever it's called
        if VALUE_Q_RE.search(query) and not ACTION_Q_RE.search(query):
            for i, c in enumerate(self.chunks):
                if scores[i] > 0 and SPEC_HEAD_RE.search(c.heading):
                    scores[i] *= 1.6
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
