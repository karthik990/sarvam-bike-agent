"""Grounded motorcycle troubleshooting agent (multi-turn, multi-question).

Pipeline per user message
  1. photo  -> gemma4 describes visible symptoms (only when a NEW photo is attached; cached)
  2. PLAN   -> one small sarvam-105b call rewrites the message into a STANDALONE English question and
              1-3 short search queries (one per distinct question). Only runs when it is needed:
              there is conversation history, the message has several questions, or it isn't English.
              A simple first question skips it (0 extra calls).
  3. RETRIEVE, per query, locally (BM25 over section-aware chunks) -> manual sections per question
  4. ANSWER -> one structured sarvam-105b call returning one "part" per question, each item with
              its page; rendered locally in a fixed layout; items not tied to a retrieved page dropped.

Conversation memory = previous STANDALONE questions + one-line answer summaries (never whole
answers, never an ever-growing topic string), so follow-ups resolve correctly without drift.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

from rag import Chunk, Index, expansion_terms

NOT_FOUND = "NOT_IN_MANUAL"
MIN_SCORE = float(os.getenv("MIN_BM25_SCORE", "0.3"))  # below this: nothing relevant, refuse locally
TOP_K = 5                # chunks retrieved per query
SECTIONS_SINGLE = 3      # sections sent for a single question
SECTIONS_PER_PART = 2    # sections per question when there are several
MAX_SECTIONS = 5         # hard cap on sections in one prompt
SECTION_CHARS = 1200     # cap per section
MAX_PARTS = 3
CONFIDENT_SCORE = float(os.getenv("CONFIDENT_SCORE", "8"))
WEAK_SCORE = float(os.getenv("WEAK_SCORE", "9"))  # first question matched weakly -> let the planner reword it  # strong match -> never show a bare refusal
PROMPT_VERSION = "p11"    # bump whenever prompts/format change -> old cached answers are ignored
PLAN_ALWAYS = os.getenv("PLAN_ALWAYS", "0") == "1"

# ---------------------------------------------------------------- prompts
PLAN_SYS = """You prepare search queries for ONE specific motorcycle owner's manual.
Given the recent conversation and the owner's LATEST message, return JSON:
{"standalone": "<the latest message rewritten as a complete, self-contained question in English; resolve 'it', 'that', 'too', 'the other one' from the conversation>",
 "queries": ["<short search query in owner's-manual wording>"],
 "language": "<BCP-47 code of the language the latest message is written in, e.g. en-IN, hi-IN, kn-IN>"}
Rules:
- Usually ONE query. Use 2-3 when the latest message asks about different things or reports different symptoms joined by 'and' (e.g. 'tyre pressure' + 'engine oil grade'; 'ABS lamp continuously on' + 'engine does not start').
- Keep the owner's symptom and context: after "my bike won't start", "the lights are dim too" -> "engine does not start lights dim weak horn".
- Use words an owner's manual uses: 'engine does not start', 'tyre pressure', 'engine oil grade', 'drive chain slackness', 'fuse blown', 'periodical maintenance', 'fuel tank capacity' (not petrol), 'kerb weight' (not how heavy), 'running in period', 'malfunction indicator lamp', 'navigation bluetooth app'.
- Do NOT add words like 'motorcycle', 'symptoms', 'troubleshooting', and do NOT guess causes or parts (no 'alternator', 'voltage').
- If the latest message starts a new topic, do NOT carry over the old topic."""

ANSWER_SYS = """You are a service advisor answering a motorcycle owner using ONLY the manual SECTIONS provided.
Return JSON: {"parts": [one object per question under QUESTIONS, same order], "not_covered": "..."}.
Each part: {"question", "found", "summary", "spec", "steps", "service_centre", "warnings"} where the lists hold {"text", "page"} items.
- found=true whenever ANY section contains information relevant to that question; give what the manual says, even if partial. found=false only if no section is about it.
- summary: one plain sentence that directly answers that question.
- Read each question by its obvious intent (e.g. 'change the engine oil grade' means change the engine oil).
- Answer ONLY what was asked. 'What is / how much / when / how often' questions: give spec and summary, and leave steps empty. 'How do I / what should I check' questions: give steps. 'What if X can't be done' questions: give the manual's advice for that case (e.g. visit a service centre), not the whole procedure again.
- Ignore sections that are not about the question (e.g. riding or gear-shifting steps for a starting problem, bulb replacement for dim lights caused by a weak battery).
- spec: ONLY numbers, limits, grades or intervals (e.g. free play 10-12 mm; replace at 10 thousand km). Actions go in steps.
- For 'how often' questions, quote the 'Maintenance item' line's schedule exactly as written (e.g. Replace at 0.5, 10, 20 thousand km); never turn a 'check level' note into a replacement interval. steps: the manual's check/procedure steps IN THE MANUAL'S ORDER, one short action each; never merge, reorder or invent.
- service_centre: when the manual says to visit a service centre. warnings: only CAUTION/WARNING about this question's task.
- page = digits of the nearest [p.N] marker ABOVE the text you used (e.g. "82").
- Per part at most 4 spec, 8 steps, 2 warnings, 2 service_centre; never repeat the same fact in two lists; each text under 18 words. No double quotes inside text.
- Never use knowledge outside the SECTIONS. not_covered: one short sentence on anything asked but not in the sections, else "".
Format example (placeholders, NOT real values; always take values from the SECTIONS): {"parts":[{"question":"<question>","found":true,"summary":"<one sentence with the manual's value>","spec":[{"text":"<item> <value with unit>","page":"<N>"}],"steps":[],"service_centre":[],"warnings":[]}],"not_covered":""}
Write all text in {lang}."""

ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "page": {"type": "string"}},
        "required": ["text", "page"], "additionalProperties": False}
PART = {
    "type": "object",
    "properties": {
        "question": {"type": "string"}, "found": {"type": "boolean"}, "summary": {"type": "string"},
        "spec": {"type": "array", "items": ITEM}, "steps": {"type": "array", "items": ITEM},
        "service_centre": {"type": "array", "items": ITEM}, "warnings": {"type": "array", "items": ITEM},
    },
    "required": ["question", "found", "summary", "spec", "steps", "service_centre", "warnings"],
    "additionalProperties": False,
}
SCHEMA = {"type": "object",
          "properties": {"parts": {"type": "array", "items": PART}, "not_covered": {"type": "string"}},
          "required": ["parts", "not_covered"], "additionalProperties": False}

HEADINGS = {  # rendered titles per language (fallback English)
    "en": ("Specification", "What to do", "Visit a service centre", "Caution", "Not covered in the manual"),
    "hi": ("विनिर्देश", "क्या करें", "सर्विस सेंटर कब जाएं", "सावधानी", "मैनुअल में नहीं है"),
}

SCRIPTS = [  # (unicode range, BCP-47)
    ((0x0900, 0x097F), "hi-IN"), ((0x0980, 0x09FF), "bn-IN"), ((0x0A00, 0x0A7F), "pa-IN"),
    ((0x0A80, 0x0AFF), "gu-IN"), ((0x0B00, 0x0B7F), "od-IN"), ((0x0B80, 0x0BFF), "ta-IN"),
    ((0x0C00, 0x0C7F), "te-IN"), ((0x0C80, 0x0CFF), "kn-IN"), ((0x0D00, 0x0D7F), "ml-IN"),
]
LANG_NAMES = {"hi-IN": "Hindi", "bn-IN": "Bengali", "pa-IN": "Punjabi", "gu-IN": "Gujarati", "od-IN": "Odia",
              "ta-IN": "Tamil", "te-IN": "Telugu", "kn-IN": "Kannada", "ml-IN": "Malayalam", "en-IN": "English"}

REFUSAL = {
    "en": "I couldn't find anything in this manual that covers that. I only answer from the uploaded manual, "
          "so please check with an authorised service centre, or describe the symptom differently.",
    "hi": "मुझे इस मैनुअल में इसके बारे में कोई जानकारी नहीं मिली। मैं केवल अपलोड किए गए मैनुअल से ही जवाब देता हूँ, "
          "कृपया अधिकृत सर्विस सेंटर से संपर्क करें या लक्षण को अलग तरह से बताएं।",
}


def detect_lang(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        for (lo, hi), code in SCRIPTS:
            if lo <= cp <= hi:
                return code
    return "en-IN"


def _refusal(lang: str) -> str:
    return REFUSAL.get(lang.split("-")[0], REFUSAL["en"])


@dataclass
class Result:
    answer: str
    found: bool
    sources: list[tuple[Chunk, float]] = field(default_factory=list)
    image_description: str | None = None
    query: str = ""                      # what was searched (shown in UI)
    language: str = "en-IN"
    warnings: list[str] = field(default_factory=list)
    api_calls: int = 0
    cached: bool = False
    reason: str = ""                     # why it refused (shown in UI)
    summary: str = ""                    # one-line summary -> conversation memory
    standalone: str = ""                 # self-contained version of the question -> conversation memory
    queries: list[str] = field(default_factory=list)
    followup: bool = False
    section_ids: list[int] = field(default_factory=list)
    debug: str = ""                      # raw model output snippet when something went wrong


# ---------------------------------------------------------------- conversation helpers
FOLLOWUP_RE = re.compile(
    r"^(and|also|so|then|ok|okay|but|what about|how about|same|next)\b|"
    r"\b(it|its|that|this|those|these|them|they|there|same|other one|previous|above|step \d+|again|too|as well|also)\b")
DANGLING_RE = re.compile(r"\b(it|its|that|this|those|these|them|they|same|other one|too|as well|step \d+)\b")


def is_dangling(question: str, history: list[dict]) -> bool:
    """True when the message can't be understood alone ('is it the same?', 'the lights are dim too').
    A message that merely starts with 'And/Also/What about' but names its own subject is NOT dangling."""
    return has_history(history) and bool(DANGLING_RE.search(question.lower()))


def split_on_and(q: str) -> list[str]:
    """'My ABS light is on and the bike won't start' -> two parts, if each part has its own subject."""
    from rag import tokenize
    parts = [x.strip(" ,.?") for x in re.split(r"\s+and\s+", q) if x.strip()]
    if 2 <= len(parts) <= 3 and all(len(tokenize(x)) >= 2 for x in parts):
        return parts
    return []


MULTI_RE = re.compile(r"\?.*\S.*\?|\band (what|how|which|when|why|where|is|are|can|should|do|does)\b|;|\balso\b", re.I)


def has_history(history: list[dict]) -> bool:
    return any(m["role"] == "assistant" for m in history)


def is_followup(question: str, history: list[dict]) -> bool:
    """Local heuristic (used for labelling and offline mode; the planner does the real resolution)."""
    return has_history(history) and bool(FOLLOWUP_RE.search(question.lower().strip()))


def looks_multi(question: str) -> bool:
    return bool(MULTI_RE.search(question)) or (" and " in question.lower() and len(question.split()) >= 8)


def context_signature(question: str, history: list[dict]) -> str:
    """Part of the cache key: with history, the meaning of a message can depend on the previous turn."""
    if not has_history(history):
        return ""
    prev = next((m for m in reversed(history) if m["role"] == "user"), {})
    return (prev.get("standalone") or prev.get("content", ""))[:200]


def _memory(history: list[dict], turns: int = 3) -> str:
    """Last N exchanges as 'Owner: <standalone question>' / 'Assistant: <one-line summary>'."""
    pairs = []
    for m in history:
        if m["role"] == "user":
            q = m.get("standalone") or m["content"]
            if m.get("img_desc"):
                q += f" (photo showed: {m['img_desc'][:120]})"
            pairs.append([q[:220], ""])
        elif pairs:
            pairs[-1][1] = (m.get("summary") or m["content"])[:200].replace("\n", " ")
    out = []
    for q, a in pairs[-turns:]:
        out.append(f"Owner: {q}")
        if a:
            out.append(f"Assistant: {a}")
    return "\n".join(out)


def cache_key(manual_hash: str, question: str, image_bytes: bytes | None) -> str:
    norm = re.sub(r"\s+", " ", question.lower()).strip(" ?.!")
    ih = hashlib.md5(image_bytes).hexdigest() if image_bytes else ""
    return hashlib.md5(f"{PROMPT_VERSION}|{manual_hash}|{norm}|{ih}".encode()).hexdigest()


# ---------------------------------------------------------------- planning
def _local_plan(q_en: str, history: list[dict]) -> dict:
    """Offline / fallback planner: split obvious multi-questions; attach the previous standalone
    question (only the previous one, never a growing chain) to pronoun follow-ups."""
    parts = [p.strip(" ,.") for p in re.split(r"\?|;|\band (?=(?:what|how|which|when|why|is|are|can|should|do)\b)", q_en)
             if len(p.strip().split()) >= 2] or [q_en]
    prev = next((m for m in reversed(history) if m["role"] == "user"), {})
    prev_q = prev.get("standalone") or prev.get("content", "")
    if is_followup(q_en, history) and prev_q:
        parts = [f"{p} {prev_q}" for p in parts]
        standalone = f"{q_en} (about: {prev_q})"
    else:
        standalone = q_en
    return {"standalone": standalone, "queries": parts[:MAX_PARTS]}


def plan(client, question: str, q_en: str, history: list[dict], lang: str, img_desc: str | None,
         force: bool = False) -> tuple[dict, int]:
    """Returns ({standalone, queries, language?}, api_calls)."""
    need = force or PLAN_ALWAYS or has_history(history) or looks_multi(q_en) or lang != "en-IN"
    if client is None or not need:
        return _local_plan(q_en, history), 0
    mem = _memory(history)
    user = (f"CONVERSATION:\n{mem}\n\n" if mem else "") + f"LATEST MESSAGE: {question}" + \
           (f"\n(English: {q_en})" if q_en != question else "") + \
           (f"\n(Photo attached showing: {img_desc})" if img_desc else "")
    try:
        data = client.chat_json([{"role": "system", "content": PLAN_SYS}, {"role": "user", "content": user}],
                                max_tokens=160)
    except Exception:
        return _local_plan(q_en, history), 1
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()][:MAX_PARTS]
    standalone = str(data.get("standalone") or "").strip()
    if not queries or not standalone:
        return _local_plan(q_en, history), 1
    return {"standalone": standalone, "queries": queries, "language": data.get("language")}, 1


# ---------------------------------------------------------------- retrieval -> sections
def build_sections(index: Index, hits, max_sections: int = SECTIONS_SINGLE, used: set | None = None) -> list[dict]:
    """Grow each top hit into its surrounding manual section: neighbouring chunks that share its
    heading, a short related sub-section on the same page (intro/spec block), or another hit on an
    adjacent page. Keeps the hit itself and stays under SECTION_CHARS."""
    chunks = index.chunks
    used = used if used is not None else set()
    sections = []
    hit_ids = {c.id for c, _ in hits}
    for c, score in sorted(hits, key=lambda x: -x[1]):
        if c.id in used:
            continue
        group, size = [c], len(c.text)

        def ok(j):
            return 0 <= j < len(chunks) and j not in used and all(j != g.id for g in group)

        lo, hi = c.id - 1, c.id + 1
        while True:
            grew = False
            for j in (hi, lo):
                if not ok(j):
                    continue
                x = chunks[j]
                same = x.heading == c.heading and abs(x.page - c.page) <= 1
                shared = set(re.findall(r"[A-Z]{4,}", x.heading)) & set(re.findall(r"[A-Z]{4,}", c.heading))
                intro = len(x.text) < 600 and x.page == c.page and bool(shared)
                related_hit = j in hit_ids and abs(x.page - c.page) <= 1
                if (same or intro or related_hit) and size + len(x.text) <= SECTION_CHARS:
                    group.append(x); size += len(x.text); grew = True
                    if j == hi: hi += 1
                    else: lo -= 1
            if not grew:
                break
        group.sort(key=lambda x: x.id)
        used.update(x.id for x in group)
        text, labels, last_h, last_l = "", [], None, None
        for x in group:
            if x.label != last_l:                      # explicit page marker -> unambiguous citations
                text += f"[p.{x.label}]\n"
                last_l = x.label
            if x.heading != last_h and x.heading:
                text += x.heading + "\n"
                last_h = x.heading
            text += x.text + "\n"
            if x.label not in labels:
                labels.append(x.label)
        same_page = next((x for x in sections if set(labels) <= set(x["labels"])), None)
        if same_page and len(same_page["text"]) + len(text) <= int(SECTION_CHARS * 1.6):
            same_page["text"] += "\n" + re.sub(r"^\[p\.[^\]]+\]\n", "", text.strip())   # same page: merge
            same_page["chunks"] = sorted(same_page["chunks"] + group, key=lambda x: x.id)
            continue
        sections.append({"labels": labels, "text": text.strip(), "score": score, "chunks": group})
        if len(sections) >= max_sections:
            break
    return sections


INTERVAL_RE = re.compile(r"\b(how often|interval|when should|when to|every how|schedule|how many (km|kms|months)|due)\b", re.I)


GENERIC_SCHED_WORDS = {"maintenanc", "interval", "servic", "schedul", "periodic", "periodical", "check", "chang",
                       "replac", "inspect", "often", "km", "month", "need", "get", "it"}


def _schedule_hit(index: Index, query: str):
    """For 'how often / when' questions: the maintenance-schedule ROW about the item asked for.
    Rows are scored individually (not whole sections) on the item's own words."""
    from rag import SCHEDULE_RE, tokenize
    subject = set(tokenize(INTERVAL_RE.sub(" ", query))) - GENERIC_SCHED_WORDS
    if not subject:
        return None
    best, best_c = 0.0, None
    for c in index.chunks:
        rows = [l for l in c.text.splitlines() if l.startswith("Maintenance item")]
        if not rows and not SCHEDULE_RE.search(c.heading):
            continue
        for row in rows or c.text.splitlines():
            name = row.split(":")[0] if row.startswith("Maintenance item") and ":" in row else row
            toks = set(tokenize(name))
            score = len(subject & toks) / len(subject) + 0.1 * len(subject & set(tokenize(row)))
            if score > best:
                best, best_c = score, c
    return (best_c, best * 10) if best_c is not None and best >= 0.5 else None


def retrieve(index: Index, queries: list[str], img_desc: str | None, context: str = ""):
    """Per-question retrieval so one question can't crowd out another. Returns (sections, hits, per_query).
    `context` (previous question, only for real follow-ups) is added as a WEAK expansion hint."""
    per_query, all_hits, used, sections = [], [], set(), []
    budget = SECTIONS_SINGLE if len(queries) == 1 else SECTIONS_PER_PART
    for i, q in enumerate(queries):
        qq = q + (f" {img_desc}" if img_desc and i == 0 else "")
        exp = (expansion_terms(qq) + " " + context).strip()
        hits = [h for h in index.search(qq, k=TOP_K, expansion=exp) if h[1] >= MIN_SCORE]
        from rag import tokenize, SYMPTOM_TERMS
        if set(tokenize(qq)) & SYMPTOM_TERMS:
            from rag import TS_RE
            ts = [h for h in hits if TS_RE.search(h[0].heading)]
            if ts and hits and hits[0] is not ts[0]:
                top = hits[0][1]
                hits = [(ts[0][0], top * 1.02)] + [h for h in hits if h is not ts[0]]
        if INTERVAL_RE.search(q):
            sh = _schedule_hit(index, q)
            if sh and all(sh[0].id != c.id for c, _ in hits):
                top = max([sc for _, sc in hits], default=sh[1])
                hits = [(sh[0], top * 1.01)] + hits      # make sure the schedule row is sent
        per_query.append((q, exp, hits))
        all_hits += hits
        secs = build_sections(index, hits, budget, used)
        for s in secs:
            s["query"] = q
        sections += secs
    return sections[:MAX_SECTIONS], all_hits, per_query


# ---------------------------------------------------------------- rendering
def _overlap_label(text: str, sources: list[tuple[str, str]]) -> str | None:
    """Grounding by content: which retrieved chunk does this item's wording come from?"""
    from rag import tokenize
    t = set(tokenize(text))
    if len(t) < 2:
        return None
    best, lab = 0.0, None
    for label, src in sources:
        o = len(t & set(tokenize(src))) / len(t)
        if o > best:
            best, lab = o, label
    return lab if best >= 0.5 else None


def resolve_page(raw, text: str, allowed: set[str], sources, pdf2label: dict) -> str | None:
    nums = re.findall(r"\d+", str(raw))
    for n in nums:                       # "82", "p.82", "[p.81/82]", "81-82"
        if n in allowed:
            return n
    for n in nums:                       # PDF page index instead of the printed one
        if pdf2label.get(n) in allowed:
            return pdf2label[n]
    return _overlap_label(text, sources)  # no usable page: keep only if the wording matches a chunk


def _part_found(pt: dict) -> bool:
    """found unless the model explicitly said false AND gave nothing; tolerate 'true'/'false' strings."""
    f = pt.get("found")
    if isinstance(f, str):
        f = f.strip().lower() not in ("false", "no", "0", "")
    has_content = any(pt.get(k) for k in ("spec", "steps", "service_centre", "warnings"))
    return bool(has_content) or (f is not False and bool(str(pt.get("summary", "")).strip())
                                 and NOT_FOUND not in str(pt.get("summary", "")))


def _normalise(data: dict) -> dict:
    """Accept the multi-part schema, common variants ('answers', 'results') and the single-answer shape."""
    for key in ("parts", "answers", "results", "items"):
        if isinstance(data.get(key), list):
            parts = [p for p in data[key] if isinstance(p, dict)]
            for p in parts:
                p["found"] = _part_found(p)
            return {"parts": parts, "not_covered": data.get("not_covered", "")}
    if any(k in data for k in ("summary", "steps", "spec")):
        p = dict(data, question=data.get("question", ""))
        p["found"] = _part_found(p)
        return {"parts": [p], "not_covered": data.get("not_covered", "")}
    return {"parts": [], "not_covered": data.get("not_covered", "")}


SPEC_TOKEN = re.compile(r"\d|\b(sae|api|jaso|psi|bar|mm|km|nm|litre|liter|ml|volt|amp|synthetic|mineral|grade|dot)\b", re.I)
CAPS = {"spec": 4, "steps": 8, "warnings": 3, "service_centre": 2}
_W = re.compile(r"[a-z0-9]+")


def _words(t: str) -> set[str]:
    return {w for w in _W.findall(t.lower()) if len(w) > 2 or w.isdigit()}


INFO_RE = re.compile(r"^\s*(what is|what are|what's|which|how much|how many|how often|when|is it|is the|are the|does|do i need)\b|"
                     r"\b(how often|interval|grade|pressure|capacity|specification)\b", re.I)
ACTION_RE = re.compile(r"\b(how do|how to|how can|what should i (do|check)|what to do|what if|adjust|fix|replace|remove|"
                       r"install|clean|can'?t|cannot|won'?t|doesn'?t|not (start|work)|dim|noise|smoke|leak|slip|"
                       r"overheat|light is on|lamp is on|stays on|blown)\b", re.I)


def is_info_question(q: str) -> bool:
    """'What is the tyre pressure / which oil / how often' -> facts only (no procedure steps).
    'When should I get it checked' is info (the interval), not a request for the procedure."""
    q = q or ""
    if re.search(r"\b(how often|interval|when should|when to)\b", q, re.I):
        return True
    return bool(INFO_RE.search(q)) and not ACTION_RE.search(q)


def relevant_warnings(warn, context: str):
    """Keep a warning only if it shares a content word with the question/answer (drops e.g. a
    gear-shifting caution attached to a starting problem)."""
    from rag import tokenize
    ctx = set(tokenize(context))
    return [(t, p) for t, p in warn if set(tokenize(t)) & ctx]


def tidy(spec, steps, warn, svc):
    """Deterministic clean-up so layout doesn't depend on the model following instructions:
    - 'Specification' holds only numbers/grades/intervals; other lines move to the right list
    - near-duplicates (same words/numbers as a fuller line) are dropped from spec/warnings/service
    - list lengths are capped; a truncated procedure points to its page"""
    moved_spec, advice = [], []
    for t, p in spec:
        if re.search(r"\b(will|may|can) (lead|cause|result|damage|affect|reduce)\b", t, re.I):
            warn.append((t, p))                                   # a consequence, not a spec
        elif re.match(r"(unwind|loosen|tighten|turn|pull|set|press|release|insert|remove|wait|push|rotate)\b", t, re.I) \
                and not re.search(r"\b(km|months?|every|interval)\b", t, re.I):
            steps.append((t, p))                                  # an action with a number, not a spec
        elif SPEC_TOKEN.search(t):
            moved_spec.append((t, p))
        elif re.search(r"service cent|dealer", t, re.I):
            svc.append((t, p))
        elif re.match(r"(do not|don't|never|avoid|caution)\b", t, re.I):
            warn.append((t, p))
        else:
            advice.append((t, p))
    spec = moved_spec
    steps = advice + [x for x in steps if x not in advice]      # direct advice first, then procedure

    def drop_subsets(lst):
        keep = []
        order = sorted(range(len(lst)), key=lambda i: -len(_words(lst[i][0])))
        chosen = []
        for i in order:
            w = _words(lst[i][0])
            if any(w <= _words(lst[j][0]) for j in chosen):
                continue
            chosen.append(i)
        for i in sorted(chosen):                       # keep original order
            keep.append(lst[i])
        return keep

    spec, warn, svc, steps = drop_subsets(spec), drop_subsets(warn), drop_subsets(svc), drop_subsets(steps)
    if len(steps) > CAPS["steps"]:
        last_page = steps[CAPS["steps"]][1]
        steps = steps[:CAPS["steps"]] + [("… remaining steps are in the manual", last_page)]
    return spec[:CAPS["spec"]], steps, warn[:CAPS["warnings"]], svc[:CAPS["service_centre"]]


def render(data: dict, lang: str, allowed: set[str], sources=None, pdf2label=None) -> tuple[str, int, int]:
    """Model JSON -> consistent markdown. Items are kept only if tied to a retrieved page.
    Returns (markdown, kept_items, dropped_items)."""
    h = HEADINGS.get(lang.split("-")[0], HEADINGS["en"])
    sources, pdf2label = sources or [], pdf2label or {}
    data = _normalise(data)
    render.dropped_raw = []
    kept = dropped = 0

    def items(part, key):
        nonlocal kept, dropped
        out = []
        for it in part.get(key) or []:
            if not isinstance(it, dict):
                it = {"text": str(it), "page": ""}
            t = str(it.get("text", "")).strip()
            if not t:
                continue
            pg = resolve_page(it.get("page", ""), t, allowed, sources, pdf2label)
            if pg is None:
                dropped += 1
                render.dropped_raw.append(str(it.get("page", ""))[:12])
                continue
            kept += 1
            out.append((t.rstrip("."), pg))
        return out

    parts = data["parts"][:MAX_PARTS]
    blocks = []

    def dedupe(lists):
        """Drop repeated lines within one answer (same wording from two pages, or the same line in
        two sections), keeping the first occurrence."""
        seen, out = set(), []
        for lst in lists:
            keep = []
            for t, p in lst:
                from rag import tokenize
                key = " ".join(sorted(set(tokenize(t)) | set(re.findall(r"\d+(?:\.\d+)?", t)))) or t.lower()
                if key in seen:
                    continue
                seen.add(key); keep.append((t, p))
            out.append(keep)
        return out

    for n, part in enumerate(parts, 1):
        spec, steps, warn, svc = tidy(*dedupe([items(part, "spec"), items(part, "steps"),
                                                items(part, "warnings"), items(part, "service_centre")]))
        if part.get("_info"):
            steps = []                                   # facts question: no procedure dump
        ctx_words = " ".join([str(part.get("question", "")), str(part.get("_q", "")), str(part.get("summary", ""))]
                             + [t for t, _ in spec + steps + svc])
        warn = relevant_warnings(warn, ctx_words)
        md = []
        if len(parts) > 1:
            md.append(f"#### {n}. {str(part.get('question', '')).strip() or 'Question ' + str(n)}")
        summ = str(part.get("summary", "")).strip()
        if summ and not (spec or steps or svc or warn):
            # uncited summary: keep only if its wording and numbers are in the retrieved manual text
            src_text = " ".join(t for _, t in sources).lower()
            nums_ok = all(n in src_text for n in re.findall(r"\d+(?:\.\d+)?", summ))
            if not (nums_ok and _overlap_label(summ, sources)):
                part = dict(part, summary="")
                summ = ""
        found = part.get("found", True) and (spec or steps or svc or warn or summ)
        if not found or NOT_FOUND in str(part.get("summary", "")):
            md.append(f"*{h[4]}.*")
            blocks.append("\n".join(md))
            continue
        if part.get("summary"):
            md.append(f"**{str(part['summary']).strip()}**")
        if spec:
            md.append(f"\n**📏 {h[0]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in spec))
        if steps:
            md.append(f"\n**🔧 {h[1]}**\n" + "\n".join(f"{i}. {t} *(p.{p})*" for i, (t, p) in enumerate(steps, 1)))
        if warn:
            md.append(f"\n**⚠️ {h[3]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in warn))
        if svc:
            md.append(f"\n**🏪 {h[2]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in svc))
        blocks.append("\n".join(md))
    nc = (data.get("not_covered") or "").strip()
    if nc and NOT_FOUND not in nc:
        blocks.append(f"*{h[4]}: {nc}*")
    return "\n\n".join(blocks), kept, dropped


def extractive_answer(sections) -> str:
    """Offline mode: show the best manual sections verbatim (0 tokens)."""
    return "**Relevant sections from your manual** (offline mode, no AI summary):\n\n" + "\n\n".join(
        f"**p.{', '.join(s['labels'])}**\n\n" + "\n".join(f"> {l}" for l in s["text"].splitlines())
        for s in sections)


def grounded_fallback(sections, per_query, lang: str) -> str | None:
    """Model refused/failed but retrieval is strong: show the manual's own words instead of a bare
    refusal (still 100% from the manual, with pages)."""
    best = max([sc for _, _, hs in per_query for _, sc in hs], default=0)
    if best < CONFIDENT_SCORE or not sections:
        return None
    note = {"hi": "मैनुअल में यह लिखा है:"}.get(lang.split("-")[0], "Here is what the manual says:")
    blocks = []
    for s in sections[:2]:
        body = "\n".join(f"> {l}" for l in s["text"].splitlines() if not l.startswith("[p."))
        blocks.append(f"**p.{', '.join(s['labels'])}**\n\n{body}")
    return f"**{note}**\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------- main entry
def answer(client, index: Index, question: str, history: list[dict], *,
           image: bytes | None = None, english_query: str | None = None, lang: str | None = None,
           image_desc_cache: dict | None = None) -> Result:
    """client=None => offline mode (no API calls at all)."""
    warnings: list[str] = []
    calls = 0
    lang = lang or detect_lang(question)

    # 1) photo -> visible symptoms (new photo only; cached by image hash)
    img_desc = None
    if image and client:
        ih = hashlib.md5(image).hexdigest()
        if image_desc_cache is not None and ih in image_desc_cache:
            img_desc = image_desc_cache[ih]
        else:
            try:
                img_desc = client.describe_image(image)
                calls += 1
                if image_desc_cache is not None:
                    image_desc_cache[ih] = img_desc
            except Exception as e:
                warnings.append(f"Image analysis unavailable ({str(e)[:100]}). Using text only.")
    elif image and not client:
        warnings.append("Offline mode: photo not analysed.")

    if not index.chunks:
        return Result("This PDF has no extractable text (it looks scanned/image-only), so it can't be searched. "
                      "Please upload a text-based PDF of the manual.", False, [], img_desc, "", lang,
                      warnings, calls, reason="no_text_in_pdf")

    # 2) plan: standalone question + one search query per distinct question
    q_en = english_query or question
    p, n = plan(client, question, q_en, history, lang, img_desc)
    calls += n
    if p.get("language") and english_query is None and lang == "en-IN" and str(p["language"]).endswith("-IN"):
        lang = p["language"] if p["language"] in LANG_NAMES else lang
    standalone, queries = p["standalone"], p["queries"]
    prev_u = next((m for m in reversed(history) if m["role"] == "user"), {})
    prev_q = (prev_u.get("standalone") or prev_u.get("content", ""))[:200]
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    if prev_q and is_dangling(question, history) and len(_words(standalone) - _words(q_en)) < 2:
        # planner echoed the message back ('the lights are dim too'): carry the previous topic
        queries = [f"{q} {prev_q}" for q in queries]
        standalone = f"{q_en.rstrip('.?! ')} (following up on: {prev_q})"
    if len(queries) == 1 and " and " in q_en.lower():
        parts = split_on_and(q_en)
        if parts:                                   # planner merged two symptoms/topics: split them
            queries = parts
    followup = is_followup(question, history) or (has_history(history) and standalone.lower() != q_en.lower())

    # 3) retrieve per question
    prev = next((m for m in reversed(history) if m["role"] == "user"), {})
    ctx_hint = (prev.get("standalone") or prev.get("content", ""))[:200] if is_dangling(question, history) else ""
    sections, hits, per_query = retrieve(index, queries, img_desc, ctx_hint)
    best0 = max([sc for _, _, hs in per_query for _, sc in hs], default=0)
    if client is not None and n == 0 and best0 < WEAK_SCORE:
        # owner's wording didn't match the manual well ('how much petrol does the tank hold'):
        # one small planner call to reword it, then keep whichever retrieval is stronger
        p2, n2 = plan(client, question, q_en, history, lang, img_desc, force=True)
        calls += n2
        s2, h2, pq2 = retrieve(index, p2["queries"], img_desc, ctx_hint)
        best2 = max([sc for _, _, hs in pq2 for _, sc in hs], default=0)
        if best2 > best0:
            sections, hits, per_query = s2, h2, pq2
            standalone, queries = p2["standalone"], p2["queries"]
    if ctx_hint:
        sections = sections[:MAX_SECTIONS - 1]           # reserve one slot for the context section
        # safety net for follow-ups: one section for "this message + previous question", in case the
        # planner's queries lost the context (e.g. 'the lights are dim too' after 'won't start')
        have = {x.id for s in sections for x in s["chunks"]}
        extra, xhits, xpq = retrieve(index, [f"{q_en} {ctx_hint}"], None)
        for sec in extra:
            if not have & {x.id for x in sec["chunks"]}:
                sec["query"] = standalone
                sections.append(sec); hits += xhits[:1]
                per_query += xpq
                break
    shown_q = " | ".join(q + (f" [+ {e}]" if e else "") for q, e, _ in per_query)
    common = dict(standalone=standalone, queries=queries, followup=followup)
    if not sections:
        best = max([s for _, _, hs in per_query for _, s in hs], default=None)
        if image and not img_desc:
            return Result("I couldn't analyse the photo (see warning above), and the question alone doesn't "
                          "describe a symptom. Please type what you see, e.g. 'the ABS warning light stays on'.",
                          False, hits, None, shown_q, lang, warnings, calls,
                          reason="photo not analysed + question has no searchable symptom", **common)
        return Result(_refusal(lang), False, hits, img_desc, shown_q, lang, warnings, calls,
                      reason=f"retrieval: nothing relevant (best score {best})", **common)
    section_ids = [x.id for s in sections for x in s["chunks"]]
    if client is None:
        return Result(extractive_answer(sections), True, hits, img_desc, shown_q, lang, warnings, 0,
                      section_ids=section_ids, **common)

    # 4) ONE structured answer call covering every question
    ctx = "\n\n".join(f"--- SECTION {i} (for: {s['query']}) ---\n{s['text']}" for i, s in enumerate(sections, 1))
    qlist = "\n".join(f"{i}. {q}" for i, q in enumerate(queries, 1)) if len(queries) > 1 else f"1. {standalone}"
    user = (f"SECTIONS:\n{ctx}\n\nOWNER'S MESSAGE: {question}\n"
            + (f"MEANING IN CONTEXT: {standalone}\n" if standalone.lower() != question.lower() else "")
            + f"QUESTIONS:\n{qlist}" + (f"\nPHOTO SHOWS: {img_desc}" if img_desc else ""))
    data = client.chat_structured([
        {"role": "system", "content": ANSWER_SYS.replace("{lang}", LANG_NAMES.get(lang, "English"))},
        {"role": "user", "content": user},
    ], SCHEMA, "manual_answer", max_tokens=1100 + 500 * (len(queries) - 1))
    calls += 1

    raw_snip = str(getattr(client, "last_raw", "") or "")[:600]
    fin = getattr(client, "last_finish", None)

    def fail(reason):
        fb = grounded_fallback(sections, per_query, lang)
        if fb:
            return Result(fb, True, hits, img_desc, shown_q, lang, warnings, calls, section_ids=section_ids,
                          reason=f"{reason}; showing the manual text instead", debug=raw_snip, **common)
        return Result(_refusal(lang), False, hits, img_desc, shown_q, lang, warnings, calls,
                      reason=reason, debug=raw_snip, **common)

    if not data:
        return fail(f"model returned no usable JSON (finish_reason={fin})")
    data = _normalise(data)
    if not any(pt.get("found") for pt in data["parts"]):
        return fail(f"model marked every question not found (finish_reason={fin})")

    # intent per part: facts-only questions get no procedure steps
    for i, pt in enumerate(data["parts"]):
        q_i = standalone if len(queries) == 1 else (queries[i] if i < len(queries) else str(pt.get("question", "")))
        pt["_q"] = q_i
        pt["_info"] = is_info_question(q_i)

    # 5) local render + citation check
    allowed = {l for s in sections for l in s["labels"]}
    srcs = [(x.label, x.text) for s in sections for x in s["chunks"]]
    pdf2label = {str(x.page): x.label for s in sections for x in s["chunks"]}
    text, kept, dropped = render(data, lang, allowed, srcs, pdf2label)
    if dropped:
        warnings.append(f"Removed {dropped} item(s) that couldn't be tied to a retrieved page "
                        f"(model cited: {', '.join(sorted(set(render.dropped_raw)))[:80]}).")
    if kept == 0:
        return fail("no answer item could be tied to a retrieved page")
    summary = " | ".join(str(pt.get("summary", "")).strip() for pt in data["parts"] if pt.get("found"))[:300]
    return Result(text, True, hits, img_desc, shown_q, lang, warnings, calls,
                  summary=summary, section_ids=section_ids, **common)
