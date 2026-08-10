import os
import io
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session

from database import SessionLocal, Base, engine, UploadHistory


load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Dataset Storytelling API")



# Database dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dataset cleaning

def clean_dataset(df: pd.DataFrame):
    """Deterministic cleaning step: report + handle missing values and duplicates."""

    report = {
        "missing_values_before": df.isnull().sum().to_dict(),
        "duplicate_rows_removed": int(df.duplicated().sum()),
    }

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Fill missing values
    # Numeric columns -> median
    # Categorical columns -> mode
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



# Correlation analysis

def compute_correlation_summary(df: pd.DataFrame):
    """Returns correlation matrix and a human-readable summary."""

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return {}, "Not enough numeric columns to compute a meaningful correlation."

    corr = numeric_df.corr()

    # Remove diagonal values
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


# AI narrative generation

def generate_ai_narrative(
    df: pd.DataFrame,
    cleaning_report: dict,
    corr_summary: str
):
    """Calls Gemini and falls back to a rule-based summary if it fails."""

    fallback = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"{cleaning_report['duplicate_rows_removed']} duplicate rows were removed. "
        f"{corr_summary}"
    )

    try:

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

Do not invent any numbers not given above.
Only describe the information provided.
"""

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        return response.text, False

    except Exception:
        return fallback, True



# Root endpoint

@app.get("/")
def root():
    return {
        "status": "API is running"
    }



# CSV upload endpoint

@app.post("/upload")
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db)
):

    # Check file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    # Read uploaded file
    contents = await file.read()

    # Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(contents))

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {e}"
        )

    # Check empty dataset
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV has no rows"
        )

    # 1. Clean dataset

    cleaned_df, cleaning_report = clean_dataset(df)

    # 2. Statistics + correlation

    describe_dict = (
        cleaned_df.describe()
        .fillna("N/A")
        .to_dict()
    )

    corr_dict, corr_summary = compute_correlation_summary(
        cleaned_df
    )

 
    # 3. AI narrative
    
    ai_narrative, used_fallback = generate_ai_narrative(
        cleaned_df,
        cleaning_report,
        corr_summary
    )

    # 4. Save to database
    
    history_entry = UploadHistory(
        filename=file.filename,
        num_rows=cleaned_df.shape[0],
        num_columns=cleaned_df.shape[1],
        fallback_summary=corr_summary,
    )

    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)

    # 5. Return response
    
    return {
        "file_name": file.filename,
        "size_bytes": len(contents),
        "num_rows": cleaned_df.shape[0],
        "num_columns": cleaned_df.shape[1],
        "column_names": cleaned_df.columns.tolist(),
        "cleaning_report": cleaning_report,
        "describe": describe_dict,
        "correlation": corr_dict,
        "fallback_summary": corr_summary,
        "ai_narrative": ai_narrative,
        "used_fallback": used_fallback,
        "history_id": history_entry.id,
    }