import os
import pandas as pd
import io
import numpy as np
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile

load_dotenv()

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()
    is_csv = file.filename.endswith('csv')

    if is_csv:
        df = pd.read_csv(io.BytesIO(contents))

        corr = df.corr(numeric_only=True)
        corr_no_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
        overall_max = corr_no_diagonal.max().max()
        max_col = corr_no_diagonal.max().idxmax()
        max_row = corr_no_diagonal[max_col].idxmax()
        summary = f"The strongest relationship found was between {max_col} and {max_row}, with a correlation of {overall_max:.2f}."

        try:
            prompt = f"""
            You are a data analyst. Write a short, clear 2-3 sentence summary of this dataset for a non-technical reader.

            Dataset facts:
            - {df.shape[0]} rows, {df.shape[1]} columns
            - Missing values: {df.isnull().sum().to_dict()}
            - Duplicate rows: {int(df.duplicated().sum())}
            - Strongest correlation: {summary}

            Do not invent any numbers not given above. Only describe what's provided.
            """

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            ai_narrative = response.text

        except Exception as e:
            ai_narrative = f"AI narrative unavailable, using fallback: {summary}"

        return {
            "file_name": file.filename,
            "size_bytes": len(contents),
            "is_csv": is_csv,
            "num_rows": df.shape[0],
            "num_colums": df.shape[1],
            "num_names": df.columns.tolist(),
            "is_null": df.isnull().sum().to_dict(),
            "duplicated_rows": int(df.duplicated().sum()),
            "describe": df.describe().fillna("N/A").to_dict(),
            "correlation": corr.fillna("N/A").to_dict(),
            "fallback_summary": summary,
            "ai_narrative": ai_narrative
        }
    else:
        return {"error": "Please upload csv file"}