# Bike Troubleshooting Agent (built on Sarvam)

Upload any bike's Owner's / Service Manual (PDF), then ask about a problem by **text, photo, or voice**.
The agent answers **only from the manual**, cites page numbers, and says so when the manual doesn't cover it.

Everything runs on Sarvam APIs:

| Step | Model | Why |
|---|---|---|
| Reasoning and grounded answers | `sarvam-105b` (the model behind Sarvam Indus) | Strong reasoning, native Indic languages |
| Photo → symptom description | `gemma4` via Sarvam `/v2/chat/completions` (beta) | `sarvam-105b` is text-only; Sarvam Vision is built for document OCR, not scene photos |
| Voice question (bonus) | `saaras:v3` STT | Handles Hindi, Kannada, Hinglish and more |
| Spoken answer (bonus) | `bulbul:v3` TTS | Reads the answer in the user's language |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your key into .env (gitignored, never commit it)
streamlit run app.py
```

**Deployed demo (Streamlit Community Cloud):** add `SARVAM_API_KEY` under App → Settings → Secrets (see `.streamlit/secrets.toml.example`). The key stays on the server and end users never see it. Since reviewers will be using your credits, the ₹ budget guard and answer cache matter here.

Tests (no API key or credits needed; they use a synthetic manual and a mocked client, and assert the number of API calls per path): `python -m pytest -q tests`

> Image input uses `gemma4`, which Sarvam gates per API key (beta). Without that access the app shows a warning and answers from the text alone.

## Keeping API spend low (credits are limited)

| Situation | Sarvam calls |
|---|---|
| Text question (English) | **1** (`sarvam-105b`, reasoning off, 4 excerpts of ≤600 chars each, reply capped at 450 tokens) |
| + new photo | **+1** (`gemma4`, image downscaled to 512 px, reply capped at 80 tokens; a repeat photo is cached and costs 0) |
| Follow-up / several questions / typed Hindi, Kannada… | **+1** small planner call (≤160 tokens out) |
| Voice question | STT only. Saaras `translate` mode returns English directly, so no extra LLM call |
| Manual doesn't cover it | **0**. Refused locally when BM25 score < `MIN_BM25_SCORE` |
| Repeated question | **0**. Answers are cached on disk in `.cache/answers.json` |
| Offline mode toggle | **0**. Shows the matching manual passages verbatim, for testing the UI and retrieval |
| TTS | Only when you click "Read aloud" |

Other savings: no chat history is sent to the model, and query expansion, language detection, retrieval and citation checks all run locally. A **session budget guard** (`SARVAM_BUDGET_INR`, default ₹100) blocks further calls once it's reached.

**Cost dashboard (sidebar):** total spend against budget, images analysed, **average cost per image**, last image's tokens and ₹ cost, how many more images fit in the remaining budget, average cost per answer, and a breakdown by call type. Each answer also shows its own cost, with the image part split out. Rates come from Sarvam's published INR price list (`pricing.py`, overridable via env). Rough estimates: a first question costs about ₹0.03–0.06, a follow-up or multi-question message about ₹0.05–0.09 (planner plus answer), and a photo about ₹0.01–0.02. ₹100 covers roughly 1,200–2,000 answers.

## Conversation memory and multi-question messages

The agent handles follow-ups ("is it the same with a pillion?", "how often should I change it?") and several questions in one message ("What's the tyre pressure and which oil should I use?").

1. **Planner step (one small call, ~₹0.01).** It runs only when needed: there is conversation history, the message has several questions, or it isn't in English. It rewrites the message as a **standalone question** ("how often should I change it?" → "How often should the engine oil be changed?") and produces **one search query per question**. A simple first question skips it.
2. **Retrieval runs separately for each question,** so one question can't crowd out another. "How often / when" questions also pull the matching row of the **periodical maintenance chart**.
3. **The answer comes in parts,** one per question, each with its own page citations.
4. **Memory holds standalone questions and one-line summaries**, never whole answers and never an ever-growing topic string. This fixed the earlier drift, where turn 5 was still searching for tyre pressure.

**Evaluation:** `python eval.py manual.pdf` checks 13 questions over 4 conversations (follow-ups, topic switches, multi-question messages, a troubleshooting chain) against the manual pages that hold the answers. Current result: **13/13**. `python eval.py manual.pdf --live` runs the same conversations on Sarvam and prints each answer, how it was understood, what it cited, and the cost (about ₹1 in total).

## Approach

```
photo ──► gemma4 (512px, "describe, don't diagnose") ─┐
voice ──► Saaras STT, translate mode → English ─────────┤
text  ──► local language detection (+ tiny translate if non-English)
                                                        ▼
            local synonym expansion ("won't start" → "engine does not start")
                                                        ▼
            BM25 over page-tagged manual chunks, top 4 ── score < threshold ──► refuse (0 calls)
                                                        ▼
            sarvam-105b (1 call) answers ONLY from excerpts, cites [p.N], or returns NOT_IN_MANUAL
                                                        ▼
            local post-check: drop citations to pages that weren't retrieved
```

Key design choices:

0. **Answer quality comes from structure, not fine-tuning.** Three things make answers consistent:
   - *Chunks follow the manual's own sections.* The text is split on the manual's ALL-CAPS headings and the repeated page headers are stripped, so one topic's caution can't leak into another's answer. Each hit is expanded to its whole section, so procedures arrive complete and in order.
   - *The model returns structured JSON* (summary, spec, steps, warnings, service centre, not covered), constrained by a JSON schema. The app renders it in the same layout every time and drops empty sections.
   - *Every item carries its own page number, and citations use the page numbers printed in the manual*, i.e. what the owner sees on the page. Any item citing a page that wasn't retrieved is dropped.

1. **The vision model describes but never diagnoses.** It only reports what it can see (e.g. "thick white smoke from the exhaust pipe"). That description becomes part of the search query. The diagnosis always comes from the manual, which keeps a capable VLM from bringing in its own general mechanical knowledge.
2. **Several layers of grounding:** (a) if retrieval finds nothing above the score threshold, the app refuses without calling the LLM; (b) the model is told to answer only from the numbered excerpts or reply `NOT_IN_MANUAL`; (c) the app checks that every page cited is one that was actually retrieved; (d) the UI shows the exact excerpts used, so the user can verify.
3. **BM25 instead of embeddings.** Manuals are short and full of specific terms ("clutch free play", "MIL", "white smoke"), and exact-term search works well on them. BM25 is also fully local, and Sarvam has no embeddings endpoint, so this keeps the whole stack on Sarvam. A local synonym map closes the vocabulary gap ("bike won't start" → "engine does not start") at zero token cost.
4. **Multilingual by default.** A user can ask in Hindi or Kannada (typed or spoken). Retrieval runs in English against the English manual, and the answer comes back in the user's language.

## Challenges and trade-offs

- **Vision on the Sarvam stack.** Sarvam's own models are text (105B) and document-OCR (Sarvam Vision). For photos I used Sarvam-hosted Gemma 4, which is in beta and needs access per key, so I added a graceful fallback to text-only answers.
- **Stopping the model from being "helpful".** LLMs like to add general advice. Fixes: a strict sentinel (`NOT_IN_MANUAL`), low temperature, "describe, don't diagnose" for vision, and citation checks after the answer.
- **Limited credits.** `sarvam-105b` thinks by default, and reasoning tokens are billed as output. I turned reasoning off, dropped an LLM query-rewrite step in favour of local expansion, and added caching, a local refusal gate, an offline mode and a budget guard (see above).
- **Scanned manuals.** Some OEM PDFs are image-only. The app detects and flags pages with no text. Next step: send those pages through Sarvam Document Intelligence (Sarvam Vision OCR, 10 pages per job) before indexing.
- **Symptoms described differently from the manual.** A photo of blue-grey smoke may match "white" or "black" smoke sections. Showing the retrieved excerpts makes this visible to the user.

## What I'd do next

Hybrid retrieval (BM25 plus embeddings), OCR for scanned pages, a small evaluation set of about 30 question/answer pairs per manual, streaming answers, and a WhatsApp front end.
