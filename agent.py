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
PROMPT_VERSION = "p7"    # bump whenever prompts/format change -> old cached answers are ignored
PLAN_ALWAYS = os.getenv("PLAN_ALWAYS", "0") == "1"

# ---------------------------------------------------------------- prompts
PLAN_SYS = """You prepare search queries for a motorcycle owner's manual.
Given the recent conversation and the owner's LATEST message, return JSON:
{"standalone": "<the latest message rewritten as a complete, self-contained English question; resolve 'it', 'that', 'the other one' etc. from the conversation>",
 "queries": ["<one short English search query per DISTINCT question in the latest message, using manual wording, e.g. 'tyre pressure', 'engine oil grade', 'clutch free play adjustment'>"],
 "language": "<BCP-47 code of the language the latest message is written in, e.g. en-IN, hi-IN, kn-IN>"}
Rules: at most 3 queries. If the latest message starts a new topic, do NOT carry over the old topic.
Include the subject (e.g. 'engine oil') in every query; never output a query like 'how often' alone."""

ANSWER_SYS = """You are a service advisor answering a motorcycle owner using ONLY the manual SECTIONS provided.
Return JSON with one entry in "parts" for EACH question listed under QUESTIONS, in the same order.
For each part:
- found=false (and lists empty) if no section answers that question.
- summary: one plain sentence that directly answers that question.
- spec: key numbers/limits stated in the manual (e.g. free play 10-12 mm).
- steps: the manual's check/procedure steps IN THE MANUAL'S ORDER, one short imperative action each. Do not merge, reorder or invent steps.
- service_centre: when the manual says to visit/contact a service centre.
- warnings: only CAUTION/WARNING text about THIS question's task.
- page = the number from the nearest [p.N] marker ABOVE the text you used, digits only (e.g. "82").
Keep it tight: per part at most 4 spec, 8 steps, 3 warnings, 2 service_centre items; each text under 20 words.
Inside text never use double quotes. Never use knowledge outside the SECTIONS.
not_covered: one short sentence on anything asked that the sections don't cover, else "".
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


# ---------------------------------------------------------------- conversation helpers
FOLLOWUP_RE = re.compile(
    r"^(and|also|so|then|ok|okay|but|what about|how about|same|next)\b|"
    r"\b(it|its|that|this|those|these|them|they|there|same|other one|previous|above|step \d+|again)\b")
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


def plan(client, question: str, q_en: str, history: list[dict], lang: str, img_desc: str | None) -> tuple[dict, int]:
    """Returns ({standalone, queries, language?}, api_calls)."""
    need = PLAN_ALWAYS or has_history(history) or looks_multi(q_en) or lang != "en-IN"
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
        sections.append({"labels": labels, "text": text.strip(), "score": score, "chunks": group})
        if len(sections) >= max_sections:
            break
    return sections


INTERVAL_RE = re.compile(r"\b(how often|interval|when should|when to|every how|schedule|how many (km|kms|months)|due)\b", re.I)


def _schedule_hit(index: Index, query: str):
    """For 'how often / when' questions: best matching row of the periodical maintenance chart."""
    subject = INTERVAL_RE.sub(" ", query)
    sched = [i for i, t in enumerate(index._flat) if "periodical maintenance" in t or "maintenance schedule" in t]
    if not sched:
        return None
    ranked = [(c, sc) for c, sc in index.search(subject, k=len(index.chunks)) if c.id in set(sched)]
    return ranked[0] if ranked else None


def retrieve(index: Index, queries: list[str], img_desc: str | None):
    """Per-question retrieval so one question can't crowd out another. Returns (sections, hits, per_query)."""
    per_query, all_hits, used, sections = [], [], set(), []
    budget = SECTIONS_SINGLE if len(queries) == 1 else SECTIONS_PER_PART
    for i, q in enumerate(queries):
        qq = q + (f" {img_desc}" if img_desc and i == 0 else "")
        exp = expansion_terms(qq)
        hits = [h for h in index.search(qq, k=TOP_K, expansion=exp) if h[1] >= MIN_SCORE]
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


def _normalise(data: dict) -> dict:
    """Accept both the multi-part schema and the older single-answer shape."""
    if isinstance(data.get("parts"), list):
        return data
    if any(k in data for k in ("summary", "steps", "spec")):
        return {"parts": [dict(data, question=data.get("question", ""))], "not_covered": data.get("not_covered", "")}
    return {"parts": [], "not_covered": data.get("not_covered", "")}


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
    for n, part in enumerate(parts, 1):
        spec, steps = items(part, "spec"), items(part, "steps")
        svc, warn = items(part, "service_centre"), items(part, "warnings")
        md = []
        if len(parts) > 1:
            md.append(f"#### {n}. {str(part.get('question', '')).strip() or 'Question ' + str(n)}")
        found = part.get("found", True) and (spec or steps or svc or warn or part.get("summary"))
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
    followup = is_followup(question, history) or (has_history(history) and standalone.lower() != q_en.lower())

    # 3) retrieve per question
    sections, hits, per_query = retrieve(index, queries, img_desc)
    shown_q = " | ".join(q + (f" [+ {e}]" if e else "") for q, e, _ in per_query)
    common = dict(standalone=standalone, queries=queries, followup=followup)
    if not sections:
        best = max([s for _, _, hs in per_query for _, s in hs], default=None)
        if image and not img_desc:
            return Result("I couldn't analyse the photo (see warning above), and the question alone doesn't "
                          "describe a symptom. Please type what you see, e.g. 'white smoke from the exhaust'.",
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
    ], SCHEMA, "manual_answer", max_tokens=700 + 350 * (len(queries) - 1))
    calls += 1

    if not data:
        fin = getattr(client, "last_finish", None)
        return Result(_refusal(lang), False, hits, img_desc, shown_q, lang, warnings, calls,
                      reason=f"model returned no usable JSON (finish_reason={fin})", **common)
    data = _normalise(data)
    if not any(pt.get("found") for pt in data["parts"] if isinstance(pt, dict)):
        return Result(_refusal(lang), False, hits, img_desc, shown_q, lang, warnings, calls,
                      reason="model found the retrieved sections unrelated — see passages below", **common)

    # 5) local render + citation check
    allowed = {l for s in sections for l in s["labels"]}
    srcs = [(x.label, x.text) for s in sections for x in s["chunks"]]
    pdf2label = {str(x.page): x.label for s in sections for x in s["chunks"]}
    text, kept, dropped = render(data, lang, allowed, srcs, pdf2label)
    if dropped:
        warnings.append(f"Removed {dropped} item(s) that couldn't be tied to a retrieved page "
                        f"(model cited: {', '.join(sorted(set(render.dropped_raw)))[:80]}).")
    if kept == 0:
        return Result(_refusal(lang), False, hits, img_desc, shown_q, lang, warnings, calls,
                      reason="no answer item could be tied to a retrieved page, so it was withheld", **common)
    summary = " | ".join(str(pt.get("summary", "")).strip() for pt in data["parts"]
                         if isinstance(pt, dict) and pt.get("found"))[:300]
    return Result(text, True, hits, img_desc, shown_q, lang, warnings, calls,
                  summary=summary, section_ids=section_ids, **common)
