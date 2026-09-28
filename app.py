import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import answer, cache_key
import pricing
from rag import Index, load_pdf
from sarvam_client import Sarvam

load_dotenv()
st.set_page_config(page_title="Bike Troubleshooter · Sarvam", page_icon="🏍️", layout="wide")

CACHE_FILE = Path(__file__).parent / ".cache" / "answers.json"


def load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def save_cache(c: dict):
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(c, ensure_ascii=False))
    except Exception:
        pass


ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("answer_cache", load_cache())   # persists across restarts: repeat demo questions cost 0
ss.setdefault("img_cache", {})
ss.setdefault("seen_audio", set())

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Setup")
    # Key is read server-side from .env (local) or Streamlit secrets (deployed) — never hardcoded,
    # never shown to end users. The input box only appears if no key is configured.
    try:
        configured = st.secrets.get("SARVAM_API_KEY", "")
    except Exception:
        configured = ""
    configured = configured or os.getenv("SARVAM_API_KEY", "")
    if configured:
        key = configured
        st.caption("✅ Sarvam API key configured")
    else:
        key = st.text_input("Sarvam API key", type="password")
    offline = st.toggle("Offline mode (0 API calls: shows manual passages only)", value=not key,
                        help="Use this while testing the UI or retrieval so you don't spend credits.")
    pdf = st.file_uploader("Owner's / Service manual (PDF)", type=["pdf"])

    if pdf is not None:
        data = pdf.getvalue()
        h = hashlib.md5(data).hexdigest()
        if ss.get("pdf_hash") != h:
            with st.spinner("Indexing manual locally…"):
                chunks, stats = load_pdf(data)
            ss.update(pdf_hash=h, index=Index(chunks), stats=stats, manual_name=pdf.name, messages=[])
        s = ss.stats
        st.success(f"{ss.manual_name}: {s['pages']} pages · {s['chunks']} chunks (indexed locally, 0 tokens)")
        if s["image_only_pages"]:
            st.warning(f"{s['image_only_pages']} scanned page(s) without text can't be searched.")
    if st.button("Clear chat"):
        ss.messages = []

    dash = st.container()   # cost dashboard is filled at the END of the run so it's always current

st.title("🏍️ Bike Troubleshooting Agent")
st.caption("Answers come **only** from the manual you upload, with page citations.")

for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=240)
        st.markdown(m["content"])

if "index" not in ss:
    st.info("Upload a bike manual PDF in the sidebar to begin.")
    st.stop()

client = None
if key and not offline:
    if ss.get("client_key") != key:
        ss.client, ss.client_key = Sarvam(key), key
    client = ss.client

# ---------------- inputs ----------------
n = len(ss.messages)
c1, c2 = st.columns(2)
with c1:
    img_file = st.file_uploader("Optional: photo of the problem", type=["png", "jpg", "jpeg", "webp"], key=f"img_{n}")
with c2:
    audio = st.audio_input("Optional: ask by voice", key=f"aud_{n}") if hasattr(st, "audio_input") else None

typed = st.chat_input("Describe the problem, e.g. 'Why is white smoke coming from my bike?'")
question, english_query, lang = typed, None, None

if not question and audio is not None and client:
    ah = hashlib.md5(audio.getvalue()).hexdigest()
    if ah not in ss.seen_audio:            # never re-transcribe the same clip on a rerun
        ss.seen_audio.add(ah)
        with st.spinner("Transcribing (Saaras)…"):
            try:
                english_query, lang = client.transcribe_to_english(audio.getvalue())
                question = english_query
            except Exception as e:
                st.error(f"Transcription failed: {e}")
if not question and img_file is not None and st.button("Ask about this photo"):
    question = "What is wrong in this photo and what should I do?"

if question:
    image = img_file.getvalue() if img_file is not None else None
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    with st.chat_message("user"):
        if image:
            st.image(image, width=240)
        st.markdown(question)
    ss.messages.append({"role": "user", "content": question, "image": image})

    ck = cache_key(ss.pdf_hash, question, image)
    with st.chat_message("assistant"):
        if client and ck in ss.answer_cache:
            r = ss.answer_cache[ck]
            st.markdown(r["answer"])
            st.caption("♻️ Cached answer — 0 API calls")
        else:
            n_before = len(client.ledger) if client else 0
            with st.spinner("Searching the manual…"):
                try:
                    res = answer(client, ss.index, question, history, image=image,
                                 english_query=english_query, lang=lang, image_desc_cache=ss.img_cache)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
            for w in res.warnings:
                st.warning(w)
            st.markdown(res.answer)
            new = client.ledger[n_before:] if client else []
            img_cost = sum(e["cost"] for e in new if e["kind"] == "image")
            total = sum(e["cost"] for e in new)
            parts = [f"Sarvam API calls: {res.api_calls}", f"cost {pricing.inr(total)}"]
            if image:
                parts.append(f"image analysis {pricing.inr(img_cost)}" + (" (cached)" if not img_cost else ""))
            st.caption(" · ".join(parts))
            r = {"cost": total, "img_cost": img_cost, "answer": res.answer, "lang": res.language, "img": res.image_description, "query": res.query,
                 "sources": [{"page": c.label, "heading": c.heading, "text": c.text[:600], "score": s}
                             for c, s in res.sources]}
            if res.reason:
                st.info(f"🔎 Why no answer: {res.reason}. Check the passages below — if they look right, "
                        "try rephrasing with words the manual uses.")
            if client and res.api_calls and res.found:   # cache only real answers, never refusals
                ss.answer_cache[ck] = r
                save_cache(ss.answer_cache)
        with st.expander("How this answer was grounded"):
            if r["img"]:
                st.markdown(f"**Photo (description only):** {r['img']}")
            st.markdown(f"**Local search query:** `{r['query'][:300]}`")
            for s in r["sources"]:
                st.markdown(f"**p.{s['page']}** {('— ' + s['heading']) if s['heading'] else ''} · score {s['score']:.2f}")
                st.text(s["text"])
    ss.messages.append({"role": "assistant", "content": r["answer"], "lang": r["lang"]})

# TTS only on explicit click (never automatic)
if client and ss.messages and ss.messages[-1]["role"] == "assistant":
    if st.button("🔊 Read last answer aloud (uses TTS credits)"):
        try:
            st.audio(client.speak(ss.messages[-1]["content"], ss.messages[-1].get("lang", "en-IN")), format="audio/wav")
        except Exception as e:
            st.caption(f"TTS unavailable: {e}")


# ---------------- cost dashboard (sidebar) ----------------
with dash:
    st.divider()
    st.subheader("💰 Cost dashboard")
    if not client:
        st.caption("Offline mode — ₹0 spent.")
    else:
        L = client.ledger
        spent = client.spent
        imgs = [e for e in L if e["kind"] == "image"]
        answers = [e for e in L if e["kind"] == "text"]
        st.progress(min(spent / client.budget, 1.0), text=f"{pricing.inr(spent)} of {pricing.inr(client.budget)} budget")
        a, b = st.columns(2)
        a.metric("Images analysed", len(imgs))
        b.metric("Avg cost / image", pricing.inr(sum(e["cost"] for e in imgs) / len(imgs)) if imgs else "—")
        a.metric("Answers", len(answers))
        b.metric("Avg cost / answer", pricing.inr(sum(e["cost"] for e in answers) / len(answers)) if answers else "—")
        if imgs:
            last = imgs[-1]
            st.caption(f"Last image: {last['in']} in + {last['out']} out tokens on {last['model']} = {pricing.inr(last['cost'])}")
            st.caption(f"≈ {int((client.budget - spent) / (sum(e['cost'] for e in imgs) / len(imgs))):,} more images fit in the remaining budget")
        if L:
            by = {}
            for e in L:
                d = by.setdefault(e["kind"], {"calls": 0, "tokens in": 0, "tokens out": 0, "cost ₹": 0.0})
                d["calls"] += 1; d["tokens in"] += e["in"]; d["tokens out"] += e["out"]; d["cost ₹"] += e["cost"]
            st.dataframe([{"type": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}}
                          for k, v in by.items()], hide_index=True)
        st.caption("Rates: Sarvam list prices (INR). Cached answers and cached photos cost ₹0.")
