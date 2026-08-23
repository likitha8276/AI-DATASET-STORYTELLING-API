import os
import io

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, UploadFile, HTTPException, Depends, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from database import SessionLocal, Base, engine, UploadHistory, User
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DataSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    history_id: int
    question: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def clean_dataset(df: pd.DataFrame):
    report = {
        "missing_values_before": {
            str(k): int(v) for k, v in df.isnull().sum().items()
        },
        "duplicate_rows_removed": int(df.duplicated().sum()),
    }

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                if pd.notna(median):
                    df[col] = df[col].fillna(median)
            else:
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])

    report["missing_values_after"] = {
        str(k): int(v) for k, v in df.isnull().sum().items()
    }

    return df, report


def compute_correlation_summary(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return {}, "Not enough numeric columns to calculate correlations."

    corr = numeric_df.corr()
    corr_no_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
    strongest = corr_no_diagonal.stack().dropna()

    if strongest.empty:
        return corr.round(2).to_dict(), "No meaningful correlation was found."

    strongest_pair = strongest.abs().idxmax()
    col1, col2 = strongest_pair[0], strongest_pair[1]
    value = corr.loc[col1, col2]

    summary = (
        f"The strongest relationship is between "
        f"{col1} and {col2}, with a correlation of {value:.2f}."
    )

    return corr.round(2).fillna("N/A").to_dict(), summary


def compute_categorical_summary(df: pd.DataFrame):
    categorical_df = df.select_dtypes(include=["object", "category", "bool"])
    summary = {}

    for col in categorical_df.columns:
        counts = categorical_df[col].value_counts()
        summary[col] = {
            "unique_values": int(categorical_df[col].nunique()),
            "top_value": str(counts.index[0]) if not counts.empty else None,
            "top_value_count": int(counts.iloc[0]) if not counts.empty else 0,
            "distribution": {str(k): int(v) for k, v in counts.items()},
        }

    return summary


def generate_group_analysis(df: pd.DataFrame):
    group_analysis = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in categorical_columns:
        groups = {}

        for value, group in df.groupby(column, dropna=False):
            stats = {"count": int(len(group))}

            for numeric_column in numeric_columns:
                values = pd.to_numeric(
                    group[numeric_column], errors="coerce"
                ).dropna()

                if len(values) > 0:
                    stats[f"avg_{numeric_column}"] = round(float(values.mean()), 2)
                    stats[f"median_{numeric_column}"] = round(float(values.median()), 2)
                    stats[f"min_{numeric_column}"] = round(float(values.min()), 2)
                    stats[f"max_{numeric_column}"] = round(float(values.max()), 2)

            groups[str(value)] = stats

        group_analysis[column] = groups

    return group_analysis


def _pick_target_column(numeric_df: pd.DataFrame):
    """Auto-select the numeric column most correlated with the rest —
    i.e. the one most 'explainable' by other columns, used as the
    regression target for residual-based outlier detection. Falls back
    to None if there's nothing suitable."""
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr().abs()
    # average correlation of each column with all others (excluding itself)
    avg_corr = (corr.sum() - 1) / (len(corr) - 1)

    if avg_corr.empty or avg_corr.max() < 0.3:
        return None  # nothing meaningfully related enough to model

    return avg_corr.idxmax()


def detect_anomalies(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return {
            "anomaly_count": 0,
            "anomaly_percentage": 0,
            "anomaly_rows": [],
            "message": "Not enough numeric columns for anomaly detection.",
        }

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median())

    model = IsolationForest(contamination="auto", random_state=42)
    predictions = model.fit_predict(numeric_df)
    scores = model.decision_function(numeric_df)  # more negative = more anomalous

    anomaly_mask = predictions == -1
    anomaly_count = int(anomaly_mask.sum())
    anomaly_percentage = round(anomaly_count / len(df) * 100, 2)

    flagged = df.loc[anomaly_mask].copy()
    flagged["_anomaly_score"] = scores[anomaly_mask]
    flagged = flagged.sort_values("_anomaly_score")

    anomaly_rows = (
        flagged.head(10).replace({np.nan: None}).to_dict(orient="records")
    )

    return {
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_percentage,
        "anomaly_rows": anomaly_rows,
        "score_range": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        },
    }


def detect_residual_outliers(df: pd.DataFrame, target_col: str = None):
    """Flags rows where a numeric column is far from what the other
    numeric columns would predict. Complements Isolation Forest by
    catching records that are inconsistent relative to expected values,
    not just large in raw magnitude."""
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return {"message": "Not enough numeric columns for residual analysis."}

    if target_col is None or target_col not in numeric_df.columns:
        target_col = _pick_target_column(numeric_df)

    if target_col is None:
        return {"message": "No suitable target column found for residual analysis."}

    feature_cols = [c for c in numeric_df.columns if c != target_col]
    if len(feature_cols) < 1:
        return {"message": "Not enough features for residual analysis."}

    X = numeric_df[feature_cols].fillna(numeric_df[feature_cols].median())
    y = numeric_df[target_col].fillna(numeric_df[target_col].median())

    model = LinearRegression().fit(X, y)
    predicted = model.predict(X)
    residuals = y - predicted

    std = residuals.std()
    if std == 0 or pd.isna(std):
        return {"message": "Residuals had no variance; skipping outlier flagging."}

    z = (residuals - residuals.mean()) / std
    outlier_mask = z.abs() > 2.5

    result = df.loc[outlier_mask].copy()
    result["_expected"] = predicted[outlier_mask].round(2)
    result["_actual"] = y[outlier_mask].round(2)
    result["_residual_zscore"] = z[outlier_mask].round(2)

    return {
        "target_column": target_col,
        "residual_outlier_count": int(outlier_mask.sum()),
        "residual_outlier_percentage": round(outlier_mask.sum() / len(df) * 100, 2),
        "rows": result.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


def generate_key_findings(
    df: pd.DataFrame,
    correlation: dict,
    group_analysis: dict,
    anomalies: dict,
):
    findings = []

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) >= 2:
        corr_df = df[numeric_columns].corr()
        pairs = []

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col1 = numeric_columns[i]
                col2 = numeric_columns[j]
                value = corr_df.loc[col1, col2]

                if not pd.isna(value):
                    pairs.append((abs(value), value, col1, col2))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, value, col1, col2 in pairs[:5]:
            if abs(value) >= 0.5:
                direction = "positive" if value > 0 else "negative"

                findings.append({
                    "type": "correlation",
                    "title": f"{col1} and {col2} show a strong {direction} relationship",
                    "description": (
                        f"{col1} and {col2} have a {direction} correlation of {value:.2f}."
                    ),
                    "evidence": {
                        "column_1": col1,
                        "column_2": col2,
                        "correlation": round(float(value), 2),
                    },
                })

    for column, groups in group_analysis.items():
        if len(groups) < 2:
            continue

        metrics = {}

        for group_name, stats in groups.items():
            for key, value in stats.items():
                if key.startswith("avg_"):
                    metrics.setdefault(key, {})
                    metrics[key][group_name] = value

        for metric, values in metrics.items():
            if len(values) < 2:
                continue

            highest_group = max(values, key=values.get)
            lowest_group = min(values, key=values.get)
            highest_value = values[highest_group]
            lowest_value = values[lowest_group]

            if highest_value == lowest_value:
                continue

            difference = highest_value - lowest_value
            metric_name = metric.replace("avg_", "")

            findings.append({
                "type": "group_difference",
                "title": f"{metric_name} varies across {column}",
                "description": (
                    f"{highest_group} has the highest average {metric_name} "
                    f"({highest_value:.2f}), while {lowest_group} has the lowest "
                    f"({lowest_value:.2f})."
                ),
                "evidence": {
                    "group_column": column,
                    "metric": metric_name,
                    "highest_group": highest_group,
                    "highest_value": highest_value,
                    "lowest_group": lowest_group,
                    "lowest_value": lowest_value,
                    "difference": round(float(difference), 2),
                },
            })

    anomaly_count = anomalies.get("anomaly_count", 0)

    if len(df) > 0 and anomaly_count > 0:
        percentage = round(anomaly_count / len(df) * 100, 2)

        findings.append({
            "type": "anomaly",
            "title": "Unusual observations detected",
            "description": (
                f"{anomaly_count} observations ({percentage}%) were flagged as "
                f"potentially unusual."
            ),
            "evidence": {"count": anomaly_count, "percentage": percentage},
        })

    return findings


def generate_ai_narrative(
    df: pd.DataFrame,
    cleaning_report: dict,
    corr_summary: str,
    anomaly_data: dict,
    residual_data: dict,
    group_analysis: dict,
    key_findings: list,
):
    fallback = (
        f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
        f"{cleaning_report['duplicate_rows_removed']} duplicate rows were removed. "
        f"{corr_summary} Isolation Forest detected "
        f"{anomaly_data.get('anomaly_count', 0)} potentially unusual observations."
    )

    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] >= 2:
            correlation_analysis = numeric_df.corr().round(2).fillna("N/A").to_dict()
        else:
            correlation_analysis = {}

        prompt = f"""
You are an expert data analyst creating a trustworthy
and meaningful data story for a non-technical reader.

Your job is to explain the evidence discovered by the
analysis engine in a clear, connected narrative.

Do not simply repeat the number of rows and columns.

Explain:
- the strongest patterns
- meaningful group differences
- relationships between variables
- unusual observations
- the overall story contained in the data

IMPORTANT STATISTICAL RULES:

1. Correlation does NOT imply causation.

2. Never say that one variable causes, drives,
determines, produces, or leads to another variable
when the evidence is based only on correlation.

3. Do not use phrases such as:
"strongest driver",
"causes",
"leads to",
"results in",
"determines",
or "increases profit"
unless explicit causal evidence is provided.

4. Use evidence-based language such as:
"is strongly associated with",
"shows a strong positive relationship with",
"tends to occur alongside",
or "is correlated with".

5. When discussing a strong correlation, remember that
the relationship is observational and does not establish
causation.

6. Do not turn correlations into business recommendations.

7. Do not invent explanations for why a relationship
exists unless the provided analysis supports them.

8. Do not assume an anomaly has a particular cause
unless the evidence supports it.

9. When comparing groups, mention actual values when
they help explain the difference.

10. Do not exaggerate small differences.

11. Clearly distinguish observed patterns from
possible interpretations.

12. Do not make business recommendations unless
the evidence directly supports them.

13. Do not say that a variable causes another variable
unless causal evidence is explicitly provided.

14. Isolation Forest detects observations that are
unusual relative to the dataset's numerical patterns.
It does not determine the cause of the anomaly.

15. Never describe anomalies as errors, exceptional
performers, high-spending records, low-spending
records, fraudulent records, or exceptional businesses
unless the provided evidence explicitly supports that
description.

16. Do not assume that an observation is unusual because
one individual value is high or low. Consider that the
anomaly detector evaluates the combined numerical
characteristics of the observation.

17. If residual analysis is provided, use it to
distinguish records that are simply large in magnitude
from records whose values are inconsistent with what
other variables would predict. Only describe a record as
"inconsistent" if residual evidence supports it.

Organize the response into these sections:

### Dataset Overview

Briefly explain what the dataset contains and its
data quality.

### Important Relationships

Explain the strongest relationships between numerical
variables.

Use actual correlation values when relevant.

Remember that these relationships are associations,
not evidence of causation.

### Group or Category Patterns

Explain meaningful differences between groups or
categories.

Mention actual values when useful.

Do not exaggerate small differences.

### Spending or Behavioral Patterns

Discuss meaningful patterns between variables when
supported by the evidence.

Do not turn correlations into causal explanations.

### Unusual Observations

Explain that the anomaly detection model identified
observations that differ from typical numerical patterns
in the dataset.

Mention the anomaly count and percentage when available.

If residual analysis flagged records, mention how many
records were both flagged by Isolation Forest and showed
values inconsistent with predictions from other columns.

Do NOT assume that anomalies are:
- errors
- high-performing companies
- low-performing companies
- high-spending companies
- low-spending companies
- fraudulent records
- exceptional businesses

unless the provided evidence explicitly supports that
conclusion.

Explain that Isolation Forest identifies unusual
observations based on their combined numerical
characteristics, but does not explain why they are
unusual.

### Overall Takeaway

Give a concise synthesis of the most important story
contained in the data.

Write several substantial paragraphs.

Make the story insightful rather than simply listing
statistics.

Do not make the response unnecessarily technical.

DATASET:

Rows:
{df.shape[0]}

Columns:
{df.shape[1]}

Column names:
{df.columns.tolist()}

CLEANING REPORT:

{cleaning_report}

STRONGEST CORRELATION:

{corr_summary}

CORRELATION ANALYSIS:

{correlation_analysis}

GROUP ANALYSIS:

{group_analysis}

KEY FINDINGS DISCOVERED BY THE ANALYSIS ENGINE:

{key_findings}

ANOMALY ANALYSIS:

{anomaly_data}

RESIDUAL OUTLIER ANALYSIS:

{residual_data}

Use the KEY FINDINGS as the primary evidence for
deciding what deserves attention.

You may connect related findings into a coherent story,
but do not introduce unsupported conclusions.

Every numerical claim must be supported by the provided
analysis.

Return only the data story.
"""

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        if not response.text:
            return fallback, True

        return response.text, False

    except Exception:
        return fallback, True


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/signup", status_code=201)
def signup(email: str, password: str, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == email).first()

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=email, hashed_password=hash_password(password))

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"id": user.id, "email": user.email}


@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(data={"sub": user.email})

    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows")

    cleaned_df, cleaning_report = clean_dataset(df)

    describe_dict = cleaned_df.describe().fillna("N/A").to_dict()

    corr_dict, corr_summary = compute_correlation_summary(cleaned_df)

    categorical_summary = compute_categorical_summary(cleaned_df)

    group_analysis = generate_group_analysis(cleaned_df)

    anomaly_data = detect_anomalies(cleaned_df)

    residual_data = detect_residual_outliers(cleaned_df)  # target auto-selected

    key_findings = generate_key_findings(
        cleaned_df, corr_dict, group_analysis, anomaly_data
    )

    ai_narrative, used_fallback = generate_ai_narrative(
        cleaned_df,
        cleaning_report,
        corr_summary,
        anomaly_data,
        residual_data,
        group_analysis,
        key_findings,
    )

    history_entry = UploadHistory(
        filename=file.filename,
        num_rows=cleaned_df.shape[0],
        num_columns=cleaned_df.shape[1],
        fallback_summary=corr_summary,
        analysis_data={
            "describe": describe_dict,
            "correlation": corr_dict,
            "cleaning_report": cleaning_report,
            "categorical_summary": categorical_summary,
            "group_analysis": group_analysis,
            "key_findings": key_findings,
            "anomalies": anomaly_data,
            "residual_outliers": residual_data,
        },
        user_id=current_user.id,
    )

    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)

    return {
        "file_name": file.filename,
        "size_bytes": len(contents),
        "num_rows": cleaned_df.shape[0],
        "num_columns": cleaned_df.shape[1],
        "column_names": cleaned_df.columns.tolist(),
        "cleaning_report": cleaning_report,
        "describe": describe_dict,
        "correlation": corr_dict,
        "categorical_summary": categorical_summary,
        "group_analysis": group_analysis,
        "key_findings": key_findings,
        "fallback_summary": corr_summary,
        "anomalies": anomaly_data,
        "residual_outliers": residual_data,
        "ai_narrative": ai_narrative,
        "used_fallback": used_fallback,
        "history_id": history_entry.id,
    }


@app.get("/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    entries = (
        db.query(UploadHistory)
        .filter(UploadHistory.user_id == current_user.id)
        .all()
    )

    return [
        {
            "id": e.id,
            "filename": e.filename,
            "num_rows": e.num_rows,
            "num_columns": e.num_columns,
            "fallback_summary": e.fallback_summary,
            "uploaded_at": e.uploaded_at,
        }
        for e in entries
    ]


@app.post("/ask")
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    history_entry = (
        db.query(UploadHistory)
        .filter(
            UploadHistory.id == request.history_id,
            UploadHistory.user_id == current_user.id,
        )
        .first()
    )

    if not history_entry:
        raise HTTPException(status_code=404, detail="Dataset history not found")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    analysis = history_entry.analysis_data

    prompt = f"""
You are a data analyst answering a user's
question about a dataset.

Use ONLY the structured dataset analysis
provided below.

Rules:

- Do not invent numbers or facts.
- Use correlations when relevant.
- Use group analysis when relevant.
- Use key findings when relevant.
- Use anomaly analysis when relevant.
- Use residual outlier analysis when relevant.
- Correlation does not imply causation.
- Do not claim that one variable causes another.
- Isolation Forest identifies unusual observations
  but does not explain their cause.
- If the available analysis is insufficient,
  say so clearly.
- Give a concise but useful answer.

STRUCTURED DATASET ANALYSIS:

{analysis}

USER QUESTION:

{request.question}
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return {
            "history_id": history_entry.id,
            "question": request.question,
            "answer": response.text,
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate an answer right now.",
        )