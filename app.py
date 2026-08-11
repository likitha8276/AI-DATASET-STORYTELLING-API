import os
import io
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from database import SessionLocal, Base, engine, UploadHistory, User
from auth import hash_password, verify_password, create_access_token, get_current_user

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Dataset Storytelling API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Data pipeline helpers (unchanged from Phase 1) ----------

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
        return {}, "Not enough numeric columns to compute a meaningful correlation."

    corr = numeric_df.corr()
    corr_no_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
    overall_max = corr_no_diagonal.max().max()

    if pd.isna(overall_max):
        return corr.fillna("N/A").to_dict(), "No significant correlation found between numeric columns."

    max_col = corr_no_diagonal.max().idxmax()
    max_row = corr_no_diagonal[max_col].idxmax()
    summary = (
        f"The strongest relationship found was between {max_col} and {max_row}, "
        f"with a correlation of {overall_max:.2f}."
    )
    return corr.fillna("N/A").to_dict(), summary


def generate_ai_narrative(df: pd.DataFrame, cleaning_report: dict, corr_summary: str):
    fallback = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"{cleaning_report['duplicate_rows_removed']} duplicate rows were removed. "
        f"{corr_summary}"
    )
    try:
        prompt = f"""
        You are a data analyst. Write a short, clear 2-3 sentence summary of this dataset for a non-technical reader.

        Dataset facts:
        - {df.shape[0]} rows, {df.shape[1]} columns (after cleaning)
        - Missing values before cleaning: {cleaning_report['missing_values_before']}
        - Duplicate rows removed: {cleaning_report['duplicate_rows_removed']}
        - Strongest correlation: {corr_summary}

        Do not invent any numbers not given above. Only describe what's provided.
        """
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        return response.text, False
    except Exception:
        return fallback, True


# ---------- Auth routes ----------

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
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm uses "username" field name; we treat it as email
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


# ---------- Core routes ----------

@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/upload")
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.endswith(".csv"):
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
    ai_narrative, used_fallback = generate_ai_narrative(cleaned_df, cleaning_report, corr_summary)

    history_entry = UploadHistory(
        filename=file.filename,
        num_rows=cleaned_df.shape[0],
        num_columns=cleaned_df.shape[1],
        fallback_summary=corr_summary,
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
        "fallback_summary": corr_summary,
        "ai_narrative": ai_narrative,
        "used_fallback": used_fallback,
        "history_id": history_entry.id,
    }


@app.get("/history")
def get_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    entries = db.query(UploadHistory).filter(UploadHistory.user_id == current_user.id).all()
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