import os
import io

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest

from database import SessionLocal, Base, engine, UploadHistory, User
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Dataset Storytelling API")


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
        "missing_values_before": df.isnull().sum().to_dict(),
        "duplicate_rows_removed": int(df.duplicated().sum()),
    }

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().any():

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

            else:
                mode = df[col].mode()

                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])

    report["missing_values_after"] = df.isnull().sum().to_dict()

    return df, report


def compute_correlation_summary(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return (
            {},
            "Not enough numeric columns to compute a meaningful correlation."
        )

    corr = numeric_df.corr()

    corr_no_diagonal = corr.where(
        ~np.eye(len(corr), dtype=bool)
    )

    overall_max = corr_no_diagonal.max().max()

    if pd.isna(overall_max):
        return (
            corr.fillna("N/A").to_dict(),
            "No significant correlation found between numeric columns."
        )

    max_col = corr_no_diagonal.max().idxmax()
    max_row = corr_no_diagonal[max_col].idxmax()

    summary = (
        f"The strongest relationship found was between "
        f"{max_col} and {max_row}, "
        f"with a correlation of {overall_max:.2f}."
    )

    return corr.fillna("N/A").to_dict(), summary


def compute_categorical_summary(df: pd.DataFrame):
    """Summarize categorical columns for dataset storytelling."""

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    summary = {}

    for col in categorical_df.columns:

        value_counts = categorical_df[col].value_counts()

        summary[col] = {
            "unique_values": int(categorical_df[col].nunique()),

            "top_value": (
                value_counts.index[0]
                if not value_counts.empty
                else None
            ),

            "top_value_count": (
                int(value_counts.iloc[0])
                if not value_counts.empty
                else 0
            ),

            "distribution": {
                str(key): int(value)
                for key, value in value_counts.items()
            },
        }

    return summary


def detect_anomalies(df: pd.DataFrame):
    """Detect unusual rows using Isolation Forest."""

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return {
            "anomaly_count": 0,
            "anomaly_percentage": 0,
            "anomaly_rows": [],
            "message": "Not enough numeric columns for anomaly detection.",
        }

    model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(numeric_df)

    anomaly_mask = predictions == -1

    anomaly_count = int(anomaly_mask.sum())

    anomaly_percentage = round(
        (anomaly_count / len(df)) * 100,
        2
    )

    anomaly_rows = df.loc[anomaly_mask].copy()

    # Convert NaN values to None so FastAPI can serialize them
    anomaly_rows = anomaly_rows.replace({np.nan: None})

    return {
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_percentage,
        "anomaly_rows": anomaly_rows.head(10).to_dict(
            orient="records"
        ),
    }


def generate_ai_narrative(
    df: pd.DataFrame,
    cleaning_report: dict,
    corr_summary: str,
    anomaly_data: dict,
):

    fallback = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"{cleaning_report['duplicate_rows_removed']} duplicate rows were removed. "
        f"{corr_summary} "
        f"Isolation Forest detected "
        f"{anomaly_data.get('anomaly_count', 0)} potential anomalies."
    )

    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        prompt = f"""
You are a data analyst.

Write a short, clear 2-3 sentence summary
of this dataset for a non-technical reader.

Dataset facts:

- {df.shape[0]} rows
- {df.shape[1]} columns
- Missing values before cleaning:
  {cleaning_report['missing_values_before']}
- Duplicate rows removed:
  {cleaning_report['duplicate_rows_removed']}
- Strongest correlation:
  {corr_summary}
- Potential anomalies detected:
  {anomaly_data.get('anomaly_count', 0)}
- Anomaly percentage:
  {anomaly_data.get('anomaly_percentage', 0)}%

Rules:

- Do not invent any numbers.
- Only describe the information provided above.
- Treat anomalies as potential unusual observations, not confirmed errors.
- Keep the explanation simple.
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


@app.post("/signup", status_code=201)
def signup(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):

    existing = db.query(User).filter(
        User.email == email
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    user = User(
        email=email,
        hashed_password=hash_password(password)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "email": user.email
    }


@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(
        User.email == form_data.username
    ).first()

    if not user or not verify_password(
        form_data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )

    token = create_access_token(
        data={"sub": user.email}
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.get("/")
def root():
    return {
        "status": "API is running"
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    contents = await file.read()

    try:
        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {e}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV has no rows"
        )

    # 1. Clean dataset
    cleaned_df, cleaning_report = clean_dataset(df)

    # 2. Descriptive statistics
    describe_dict = (
        cleaned_df.describe()
        .fillna("N/A")
        .to_dict()
    )

    # 3. Correlation analysis
    corr_dict, corr_summary = compute_correlation_summary(
        cleaned_df
    )

    # 4. Categorical analysis
    categorical_summary = compute_categorical_summary(
        cleaned_df
    )

    # 5. Anomaly detection
    anomaly_data = detect_anomalies(
        cleaned_df
    )

    # 6. AI narrative
    ai_narrative, used_fallback = generate_ai_narrative(
        cleaned_df,
        cleaning_report,
        corr_summary,
        anomaly_data,
    )

    # 7. Save analysis
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
            "anomalies": anomaly_data,
        },

        user_id=current_user.id,
    )

    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)

    # 8. Return response
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

        "fallback_summary": corr_summary,

        "anomalies": anomaly_data,

        "ai_narrative": ai_narrative,

        "used_fallback": used_fallback,

        "history_id": history_entry.id,
    }


@app.get("/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    entries = (
        db.query(UploadHistory)
        .filter(
            UploadHistory.user_id == current_user.id
        )
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
        raise HTTPException(
            status_code=404,
            detail="Dataset history not found"
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    analysis = history_entry.analysis_data

    prompt = f"""
You are a data analyst answering a user's question about a dataset.

Use ONLY the structured dataset analysis provided below.

Rules:

- Do not invent numbers, categories, or facts.
- Use the statistics, correlations, categorical summaries,
  and anomaly information when relevant.
- If the available analysis is not sufficient to answer the question,
  clearly say so.
- Give a short, clear answer for a non-technical reader.
- If the question asks about categories, use categorical_summary.
- If the question asks about unusual or abnormal records, use anomalies.
- If the question asks about averages, minimums, maximums,
  or distributions, use describe.
- If the question asks about relationships between numeric columns,
  use correlation.

STRUCTURED DATASET ANALYSIS:

{analysis}

USER QUESTION:

{request.question}
"""

    try:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return {
            "history_id": history_entry.id,
            "question": request.question,
            "answer": response.text,
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate an answer right now."
        )