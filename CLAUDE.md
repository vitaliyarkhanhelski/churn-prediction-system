# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Diploma project ("Projekt Dyplomowy"): a bank customer churn prediction system with an AI email
agent. A customer manager uploads a CSV, a Voting Classifier scores churn risk, SHAP explains
*why*, and a LangGraph agent drafts + sends a personalized email per at-risk customer, augmented
with RAG-retrieved marketing campaigns.

The app and all in-code comments/docstrings/UI strings are in **Polish** — keep new code
consistent with that.

## Commands

```bash
# Install deps (from repo root)
pip install -r requirements.txt

# Run the Streamlit app (must run from main/ — code uses paths relative to that dir)
cd main
streamlit run app.py          # served at http://localhost:8501

# Run the prediction pipeline standalone (also from main/)
cd main
python predict_churn.py       # reads data/, writes results/dzisiejsze_cele_ratunkowe.csv
```

There is no test suite, linter, or build step configured in this repo.

### Local LLM (optional, for the "Bielik" backend)

```bash
ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
ollama serve       # must be running before selecting Bielik in the UI
```

### Required `main/.env`

```
SENDER_EMAIL=...
SENDER_APP_PASSWORD=...   # Gmail App Password, not the account password
OPENAI_API_KEY=sk-...
```

## Architecture

Everything lives in `main/`; imports are relative to that directory (`app.py` inserts its own
dir onto `sys.path`), so scripts must be run with `main/` as the working directory.

**Pipeline:** `app.py` (Streamlit UI) → `predict_churn.py` (ML + SHAP) → `langgraph_agent.py`
(AI email generation + send) → `email_service.py` (SMTP).

- **`predict_churn.py`** — `run_prediction(df, strategy)` loads two *separate* pickled models:
  `models/voting_model.pkl` (VotingClassifier: XGBoost + SVM + LogisticRegression) produces the
  churn probability used for the actual decision, while `models/xgboost_for_shap.pkl` is used
  only to compute SHAP values for explainability (its own preprocessing pipeline, via
  `feature_in_XGB`). Marketing-action flag is `churn_probability >= threshold`, where threshold
  is one of two hardcoded business strategies: `THRESHOLD_PRECYZYJNY = 0.79` (precision-first) or
  `THRESHOLD_SZEROKI = 0.53` (recall/F1-first). Returns a merged DataFrame (customer data +
  `churn_probability` + `marketing_action` + one `shap_<feature>` column per feature) plus the
  raw SHAP `Explanation` object (needed for waterfall plots in the UI).

- **`utils.py`** — `REQUIRED_COLUMNS` is the CSV schema contract enforced by `app.py` on upload.
  `feature_in_XGB`/`feature_in_SVC`/`feature_in_LR` are per-model feature engineering functions —
  each trained sub-model of the VotingClassifier expects a different feature set/transform, so
  don't assume one shared preprocessing path.

- **`langgraph_agent.py`** — a LangGraph `StateGraph` (`StanGrafu` state) run once per at-risk
  customer, fully async and fanned out concurrently:
  `explain_shap → Agent_RAG ⇄ tools(marketing_search) → Summarize → ChooseCampaign →
  prepare_mail → send_and_log`.
  - `explain_shap` turns that customer's SHAP values into a Polish risk narrative via
    `_shap_to_text()`. **This function is fact-critical**: it takes both the SHAP values *and*
    the customer's raw features. One-hot columns (`shap_country_*`, `shap_gender_*`) describe a
    *feature*, not a fact — `shap_country_Germany` is non-zero even for a French customer — so
    only the column matching the customer's actual value is emitted (via `ONE_HOT_LABELS`).
    Numeric features get their real value plus distribution context (`FEATURE_QUARTILES`,
    `PRODUCTS_NUMBER_CONTEXT`) so the LLM describes rather than guesses. Before this, emails
    stated the wrong country, listed both genders, and invented values.
  - `Agent_RAG` is a tool-calling agent that searches a persisted Chroma vector store
    (`Marketing_cam_test/`, built from `rag/Marketing_cam/*.docx`); loops through `tools` until
    it stops calling tools. `marketing_search` retrieves `k=40` **chunks** (each campaign doc is
    ~20-25 chunks), then dedupes by source and keeps the top `MAX_CAMPAIGN_CANDIDATES` (4)
    *distinct* campaigns — a narrow query used to return 4 chunks of one campaign, leaving
    nothing to choose between.
  - `Summarize` re-loads the `.docx` files found via RAG and summarizes them per-chunk then
    overall, caching each campaign's summary process-wide (`_summary_cache`, with per-file locks)
    since the same ~10 static docs recur across customers. Returns both the display text
    (`from_summarizer`) and a structured, retrieval-ordered `campaigns` list.
  - `ChooseCampaign` picks exactly one campaign from the candidates and is **always forced onto
    OpenAI**, regardless of the selected backend. Selection used to be an implicit side-effect of
    the drafting prompt; Bielik ignored it (described all candidates verbatim) and gpt-4o-mini
    contradicted itself. Returns `chosen_campaign_source` plus `campaign_choice_reasoning`, which
    is written into the email log for auditability.
  - `prepare_mail` drafts the email body/subject around the single already-chosen campaign, and
    builds the PDF attachment path **directly from `chosen_campaign_source`**. Do not reintroduce
    matching campaign titles against the generated email text — that approach broke three
    separate ways (dropped "Kampania" prefix, NFC/NFD Unicode mismatch, reworded titles).
  - `send_and_log` sends via `email_service.send_emails` and always writes a log file to
    `emails/<timestamp>_<customer_id>.txt` (SHAP summary, candidate campaigns, choice reasoning,
    delivery status, full email body) regardless of send success/failure.
  - **Model backend**: `model_backend` is `"openai"` (`OPENAI_MODEL`, `asyncio.Semaphore(2)`) or
    `"ollama"` (local Bielik, `Semaphore(1)` — sequential). `Agent_RAG`, `Summarize` and
    `ChooseCampaign` always force OpenAI regardless of the chosen backend; only `explain_shap`
    and `prepare_mail` follow the user's UI selection. Note `OpenAIEmbeddings` is used for the
    vector store too, so selecting Bielik does **not** mean a fully local pipeline.
  - **Rate limits**: gpt-4o's TPM ceiling is easy to hit (several LLM calls per customer). Three
    mitigations, all load-bearing: `_retryable()` (exponential backoff on `RateLimitError`), the
    summary cache, and `Semaphore(2)`. `_retryable()` must be applied *after* `bind_tools()` —
    `with_retry()` on a raw model strips tool binding.
  - Entry point: `run_agent(df_flagged, to_email, model_backend, run_id)` builds one task per row
    and gathers them; called from `app.py` inside a background `threading.Thread`. Per-customer
    failures are logged to `emails/<timestamp>_<customer_id>_FAILED.txt` with a traceback
    (`_log_failure`) — without this they vanish silently, since `asyncio.gather` collects
    exceptions and `app.py`'s thread wrapper swallows them.
  - **Progress store**: `_push_progress`/`get_progress` keep per-run send status in a
    lock-guarded process-global dict, polled by the UI. A background thread cannot safely write
    to `st.session_state`, hence the module-level store.

- **`prompts.py`** — all LLM prompt templates plus the SHAP labelling data used by
  `_shap_to_text`: `FEATURE_LABELS` (numeric features), `ONE_HOT_LABELS` (keyed by
  `(column, actual_value)`), `FEATURE_VALUE_FORMATTERS`, and the distribution context
  (`FEATURE_QUARTILES`, `PRODUCTS_NUMBER_CONTEXT`) measured from the 10k-row training set.
  Prompt rules here are mostly scar tissue from observed model failures — read the comments
  before "simplifying" them.

- **`email_service.py`** — plain `smtplib` over Gmail SMTP (SSL, port 465). Requires
  `SENDER_EMAIL`/`SENDER_APP_PASSWORD` env vars (raises if missing). Builds one `MIMEMultipart`
  per email with optional PDF attachments; returns a list of per-recipient error strings rather
  than raising, so callers must check the return value.

- **`app.py`** — Streamlit UI and orchestration only; no business logic. Notable patterns:
  - Uploaded CSV is validated against `REQUIRED_COLUMNS` and extra columns are dropped.
  - Results (`df_final`, `shap_vals`, etc.) are stashed in `st.session_state` so they survive
    reruns; every run is also persisted to `results/churn_<timestamp>.csv`.
  - "Send emails" uses an on_click callback to set a `sending_emails` flag, then a background
    daemon thread calls `run_agent` so the UI thread returns immediately; an `st.empty()` slot
    swaps between the button and a "sending" state to avoid duplicate renders, and a
    `pending_toast` session key survives the following `st.rerun()` to show a toast once.
  - Section "6. Status wysyłki e-maili" renders live send progress from `get_progress(run_id)`
    inside an `@st.fragment(run_every="10s")`, so only that table refreshes. The `run_id` is the
    results CSV filename (already unique per analysis run).

- **`bank_customer_churn/`** — a separate, standalone offline notebook environment for training
  the VotingClassifier and the SHAP XGBoost model from the Kaggle dataset (ROC-AUC 0.86). It is
  git-ignored and independent of `main/`; trained `.pkl` artifacts must be copied manually into
  `main/models/` to be picked up by `predict_churn.py`.

- **`rag/`** — source material for the marketing-campaign RAG: `Marketing_cam/*.docx` is what
  gets embedded/searched; `Files_to_attach/*.pdf` are the matching attachments sent to customers.
  Filenames must correspond 1:1 (`.docx` stem ↔ `.pdf` stem) since `prepare_mail` derives the PDF
  path from the docx source path.

- **`emails/`** and **`results/`** — runtime output logs (one `.txt` per sent email, plus
  `*_FAILED.txt` with a traceback for customers whose pipeline threw; one `.csv` per analysis
  run). Not code, but the primary debugging surface: each email log records the SHAP narrative,
  all retrieved campaign candidates, and the reasoning behind the campaign choice, which is how
  output-quality regressions get diagnosed.

## Marketing strategies (business logic, not just UI copy)

Two selectable thresholds in `predict_churn.py`, surfaced in the sidebar:
- 🎯 Precyzyjny (0.79) — precision > 89%, fewer contacts, low false-positive rate.
- 🔍 Szeroki zasięg (0.53) — max F1 (0.639), more customers saved but more contacts/false alarms.
