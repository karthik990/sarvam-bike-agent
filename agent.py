"""Token-frugal troubleshooting agent.

Per question, the Sarvam API is called AT MOST:
  - 0x  if the answer is cached, the manual has no match (local refusal), or offline mode is on
  - 1x  sarvam-105b for the grounded answer (reasoning off, ~4 short excerpts, 450-token cap)
  - +1x gemma4 only if a NEW photo is attached (80-token cap, image downscaled, cached by hash)
  - +1x tiny translate call only if the question is TYPED in a non-Latin script
Everything else (language detection, query expansion, retrieval, refusal gate, citation check)
runs locally.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

from rag import Chunk, Index, expansion_terms

NOT_FOUND = "NOT_IN_MANUAL"
MIN_SCORE = float(os.getenv("MIN_BM25_SCORE", "0.3"))  # below this: refuse locally, no API call
TOP_K = 5            # chunks retrieved
MAX_SECTIONS = 3     # sections sent to the model
SECTION_CHARS = 1300 # cap per section (keeps prompt ~900 tokens)
PROMPT_VERSION = "p4"  # bump whenever the prompt/format changes -> old cached answers are ignored

ANSWER_SYS = """You are a service advisor answering a motorcycle owner using ONLY the manual SECTIONS provided.
Return JSON. Rules:
- found=false (and all lists empty) if no section is about the question.
- summary: one plain sentence that directly answers the question.
- spec: key numbers/limits stated in the manual (e.g. free play 10-12 mm). Empty if none.
- steps: the manual's check/procedure steps IN THE MANUAL'S ORDER, one short imperative action each, same meaning as the manual. Do not merge, reorder or invent steps.
- service_centre: when the manual says to visit/contact a service centre.
- warnings: only CAUTION/WARNING text that concerns THIS task. Ignore warnings about other topics.
- not_covered: one short sentence on what the question asks that the sections don't cover, else "".
- page = the number from the nearest [p.N] marker ABOVE the text you used, digits only (e.g. "82").
- Keep it tight: at most 4 spec, 8 steps, 3 warnings, 2 service_centre items.
- Never use knowledge outside the sections. Write the text fields in {lang}."""

ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "page": {"type": "string"}},
        "required": ["text", "page"], "additionalProperties": False}
SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "summary": {"type": "string"},
        "spec": {"type": "array", "items": ITEM},
        "steps": {"type": "array", "items": ITEM},
        "service_centre": {"type": "array", "items": ITEM},
        "warnings": {"type": "array", "items": ITEM},
        "not_covered": {"type": "string"},
    },
    "required": ["found", "summary", "spec", "steps", "service_centre", "warnings", "not_covered"],
    "additionalProperties": False,
}

HEADINGS = {  # rendered section titles per language (fallback English)
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
    query: str = ""
    language: str = "en-IN"
    warnings: list[str] = field(default_factory=list)
    api_calls: int = 0
    cached: bool = False
    reason: str = ""   # why it refused (shown in UI for debugging)


def cache_key(manual_hash: str, question: str, image_bytes: bytes | None) -> str:
    norm = re.sub(r"\s+", " ", question.lower()).strip(" ?.!")
    ih = hashlib.md5(image_bytes).hexdigest() if image_bytes else ""
    return hashlib.md5(f"{PROMPT_VERSION}|{manual_hash}|{norm}|{ih}".encode()).hexdigest()


def build_sections(index: Index, hits) -> list[dict]:
    """Grow each top hit into its surrounding manual section: add the neighbouring chunks that share
    its heading (and a tiny intro/spec block right before it), always keeping the hit itself and
    staying under SECTION_CHARS. Gives the model whole procedures in the manual's order."""
    chunks, used, sections = index.chunks, set(), []
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
                # short sub-section on the same page (intro, spec block) belongs with this one
                shared = set(re.findall(r"[A-Z]{4,}", x.heading)) & set(re.findall(r"[A-Z]{4,}", c.heading))
                intro = len(x.text) < 600 and x.page == c.page and bool(shared)
                related_hit = j in hit_ids and abs(x.page - c.page) <= 1   # e.g. spec block next to procedure
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
        if len(sections) >= MAX_SECTIONS:
            break
    return sections  # best-scoring section first


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
    for n in nums:                       # model used the PDF page index instead of the printed one
        if pdf2label.get(n) in allowed:
            return pdf2label[n]
    return _overlap_label(text, sources)  # no usable page: accept only if the wording matches a chunk


def render(data: dict, lang: str, allowed: set[str], sources=None, pdf2label=None) -> tuple[str, int, int]:
    """Turn the model's JSON into a consistent, readable answer. Items are kept only if they can be
    tied to a retrieved page. Returns (markdown, kept_items, dropped_items)."""
    h = HEADINGS.get(lang.split("-")[0], HEADINGS["en"])
    kept = dropped = 0
    sources, pdf2label = sources or [], pdf2label or {}
    render.dropped_raw = []

    def items(key):
        nonlocal kept, dropped
        out = []
        for it in data.get(key) or []:
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

    spec, steps, svc, warn = items("spec"), items("steps"), items("service_centre"), items("warnings")
    md = [f"**{data.get('summary', '').strip()}**"] if data.get("summary") else []
    if spec:
        md.append(f"\n**📏 {h[0]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in spec))
    if steps:
        md.append(f"\n**🔧 {h[1]}**\n" + "\n".join(f"{i}. {t} *(p.{p})*" for i, (t, p) in enumerate(steps, 1)))
    if warn:
        md.append(f"\n**⚠️ {h[3]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in warn))
    if svc:
        md.append(f"\n**🏪 {h[2]}**\n" + "\n".join(f"- {t} *(p.{p})*" for t, p in svc))
    nc = (data.get("not_covered") or "").strip()
    if nc:
        md.append(f"\n*{h[4]}: {nc}*")
    return "\n".join(md), kept, dropped


def extractive_answer(sections) -> str:
    """Offline mode: show the best manual sections verbatim (0 tokens)."""
    return "**Relevant sections from your manual** (offline mode, no AI summary):\n\n" + "\n\n".join(
        f"**p.{', '.join(s['labels'])}**\n\n" + "\n".join(f"> {l}" for l in s["text"].splitlines())
        for s in sections)


def answer(client, index: Index, question: str, history: list[dict], *,
           image: bytes | None = None, english_query: str | None = None, lang: str | None = None,
           image_desc_cache: dict | None = None) -> Result:
    """client=None => offline mode (no API calls at all)."""
    warnings: list[str] = []
    calls = 0
    lang = lang or detect_lang(question)

    # 1) photo -> visible symptoms (only if a photo is attached; cached by image hash)
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

    # 2) build an English retrieval query locally (translate only for typed non-English)
    q_en = english_query or question
    if english_query is None and lang != "en-IN" and client:
        try:
            q_en = client.to_english_keywords(question)
            calls += 1
        except Exception as e:
            warnings.append(f"Translation failed ({str(e)[:80]}).")
    # short follow-ups ("what about the chain?") borrow the previous question for retrieval only
    prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    refers_back = re.search(r"\b(it|that|this|those|them|same|other|also|again|what about|and)\b", q_en.lower())
    if prev_user and len(q_en.split()) <= 5 and refers_back:
        q_en = f"{q_en} {prev_user[:200]}"
    base_q = q_en + (f" {img_desc}" if img_desc else "")
    exp = expansion_terms(base_q)
    search_q = base_q + (f"  [+ {exp}]" if exp else "")

    # 3) retrieve + local refusal gate
    if not index.chunks:
        return Result("This PDF has no extractable text (it looks scanned/image-only), so it can't be searched. "
                      "Please upload a text-based PDF of the manual.", False, [], img_desc, search_q, lang,
                      warnings, calls, reason="no_text_in_pdf")
    raw = index.search(base_q, k=TOP_K, expansion=exp)
    hits = [h for h in raw if h[1] >= MIN_SCORE]
    if not hits:
        best = f"{raw[0][1]:.2f}" if raw else "no term overlap"
        if image and not img_desc:
            return Result("I couldn't analyse the photo (see warning above), and the question alone doesn't "
                          "describe a symptom. Please type what you see, e.g. 'white smoke from the exhaust'.",
                          False, raw, None, search_q, lang, warnings, calls,
                          reason="photo not analysed + question has no searchable symptom")
        return Result(_refusal(lang), False, raw, img_desc, search_q, lang, warnings, calls,
                      reason=f"retrieval: best match score {best} < threshold {MIN_SCORE}")
    hits.sort(key=lambda x: x[0].page)

    sections = build_sections(index, hits)
    if client is None:
        return Result(extractive_answer(sections), True, hits, None, search_q, lang, warnings, 0)

    # 4) ONE structured call: model returns JSON, we render it locally in a fixed layout
    ctx = "\n\n".join(f"--- SECTION {i} ---\n{s['text']}" for i, s in enumerate(sections, 1))
    user = f"SECTIONS:\n{ctx}\n\nQUESTION: {question}" + (f"\nPHOTO SHOWS: {img_desc}" if img_desc else "")
    data = client.chat_structured([
        {"role": "system", "content": ANSWER_SYS.replace("{lang}", LANG_NAMES.get(lang, "English"))},
        {"role": "user", "content": user},
    ], SCHEMA, "manual_answer")
    calls += 1

    if not data:
        fin = getattr(client, "last_finish", None)
        return Result(_refusal(lang), False, hits, img_desc, search_q, lang, warnings, calls,
                      reason=f"model returned no usable JSON (finish_reason={fin})")
    if not data.get("found") or NOT_FOUND in str(data.get("summary", "")):
        return Result(_refusal(lang), False, hits, img_desc, search_q, lang, warnings, calls,
                      reason="model found the retrieved sections unrelated — see passages below")

    # 5) local render + citation check (items citing non-retrieved pages are dropped)
    allowed = {l for s in sections for l in s["labels"]}
    srcs = [(x.label, x.text) for s in sections for x in s["chunks"]]
    pdf2label = {str(x.page): x.label for s in sections for x in s["chunks"]}
    text, kept, dropped = render(data, lang, allowed, srcs, pdf2label)
    if dropped:
        warnings.append(f"Removed {dropped} item(s) that couldn't be tied to a retrieved page "
                        f"(model cited: {', '.join(sorted(set(render.dropped_raw)))[:80]}).")
    if kept == 0:
        return Result(_refusal(lang), False, hits, img_desc, search_q, lang, warnings, calls,
                      reason="no answer item could be tied to a retrieved page, so it was withheld")
    return Result(text, True, hits, img_desc, search_q, lang, warnings, calls)
