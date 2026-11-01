# DataSense: AI Dataset Storytelling API

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-orange)
![Gemini](https://img.shields.io/badge/Gemini-API-4285F4)
![Auth](https://img.shields.io/badge/Auth-JWT-black)

Upload a CSV, get back a cleaning report, statistics, anomaly flags, and a plain-English narrative.

**Core idea:** the LLM never sees raw rows. Cleaning, stats, correlations, and anomaly detection are deterministic Python. Gemini only receives the computed summary and writes prose. Remove the Gemini call and the API still returns a full analysis with a rule-based summary.

<!-- Add a screenshot or GIF of the /docs page or Streamlit dashboard here:
![demo](docs/demo.png) -->

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Example](#example)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

## Features

- **Transparent cleaning:** drops duplicates, imputes missing values (median for numeric, mode for categorical), and reports before/after counts instead of cleaning silently
- **Stats and correlations:** `describe()` plus strongest-pair correlation, guarded for edge cases (fewer than 2 numeric columns, all-NaN matrix)
- **Dual anomaly detection:** Isolation Forest plus linear-regression residuals, ranked by severity via `decision_function()`
- **LLM narrative with fallback:** Gemini turns the summary into prose; any failure returns a rule-based summary and sets `used_fallback: true`
- **`/ask` endpoint:** plain-English questions about your data, grounded in computed stats
- **JWT auth** with per-user upload history
- **Streamlit dashboard** on top of the API

## Architecture

```
Client (Swagger / Streamlit)
        |
        v
FastAPI
 |-- Auth: /signup, /login (JWT, bcrypt)
 |-- API: /upload, /history, /ask
 |
 |-- Core engine (local, deterministic)
 |     clean -> describe -> correlate -> anomaly detection
 |
 |-- Gemini (summary in, prose out)
 |     on failure -> rule-based fallback
 |
 v
SQLite via SQLAlchemy (users, upload history)
```

```
.
├── app.py            # routes + analysis pipeline
├── auth.py           # hashing, JWT, get_current_user
├── database.py       # engine, session, User + UploadHistory models
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start

```bash
git clone https://github.com/likitha8276/ai-dataset-storytelling-api.git
cd ai-dataset-storytelling-api
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
cp .env.example .env              # then fill in your values
```

`.env`:

```
DATABASE_URL=sqlite:///./dataset_storytelling.db
GEMINI_API_KEY=your_key
SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
```

Run:

```bash
python -m fastapi dev app.py
```

Open http://127.0.0.1:8000/docs, then: `/signup` -> `/login` -> copy `access_token` -> **Authorize** -> `/upload`.

Dashboard: `streamlit run <your_dashboard_file>.py`

## API reference

| Method | Route | Auth | Purpose |
|---|---|---|---|
| POST | `/signup` | no | Create account |
| POST | `/login` | no | OAuth2 form login, returns JWT |
| POST | `/upload` | yes | Upload CSV, get full analysis |
| GET | `/history` | yes | Your past uploads |
| POST | `/ask` | yes | Ask a question about your data |

## Example

```bash
# log in (username = email)
curl -X POST http://127.0.0.1:8000/login \
  -d "username=you@example.com&password=yourpassword"

# upload
curl -X POST http://127.0.0.1:8000/upload \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@data.csv"
```

Abridged response from a real 9,555-row, 14-column test file:

```json
{
  "num_rows": 9555,
  "num_columns": 14,
  "cleaning_report": {
    "missing_values_before": { "Data_value": 45, "Suppressed": 9540, "Series_title_5": 9555 },
    "duplicate_rows_removed": 0
  },
  "fallback_summary": "The strongest relationship found was between Period and Data_value, with a correlation of 0.11.",
  "anomalies": { "anomaly_count": 3242, "anomaly_percentage": 33.93 },
  "ai_narrative": "This dataset contains 9,555 rows and 14 columns with no duplicate rows ...",
  "used_fallback": false,
  "history_id": 3
}
```

## Design decisions

- **Summary to the LLM, not rows:** cheaper, no token ceiling on dataset size, no raw-data leakage, and every number in the narrative traces to computed output
- **Fallback on any Gemini exception:** the API stays up when the AI service doesn't, and `used_fallback` makes it visible
- **Median/mode imputation:** simple and explainable, though not ideal for skewed or multimodal columns
- **SQLite for local dev:** zero config; moving to PostgreSQL is a `DATABASE_URL` change because everything goes through SQLAlchemy
- **Per-request DB sessions** via `Depends(get_db)` rather than one shared session

**Built myself:** cleaning pipeline and report, correlation guards, dual anomaly detection and ranking, fallback, auth and per-user history, prompt construction.
**Integrated:** Gemini, scikit-learn's Isolation Forest, bcrypt/JWT libraries.

## Limitations

- Fully-null columns are reported but not dropped
- 33.93% anomaly rate on the test file suggests `contamination` needs per-dataset tuning
- Frontend misreads `$` in narratives as LaTeX math delimiters
- No rate limiting, file size cap, or refresh tokens
- SQLite is single-writer, so not for concurrent production load

## Roadmap

- [ ] Tune or expose `contamination`; drop or flag fully-null columns
- [ ] Redis cache for repeated `/ask` queries
- [ ] PostgreSQL + Alembic migrations
- [ ] Pytest coverage for cleaning and correlation
- [ ] Dockerfile
- [ ] Pydantic body for `/signup` instead of query params
