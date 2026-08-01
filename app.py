import pandas as pd
import io
import numpy as np
from fastapi import FastAPI , UploadFile
app = FastAPI()
@app.post("/upload")
async def upload_file(file : UploadFile):
    print("DEBUG: NEW CODE IS RUNNING")
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

     return {
        "TESTING_123": file.filename,
        "size_bytes": len(contents),
        "is_csv" : is_csv,
        "num_rows": df.shape[0],
        "num_colums" : df.shape[1],
        "num_names": df.columns.tolist(),
        "is_null": df.isnull().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "describe": df.describe().fillna("N/A").to_dict(),
        "correlation": df.corr(numeric_only=True).fillna("N/A").to_dict(),
        "fallback_summary": summary
    }
    else :
       return {"error":"Please upload csv file"}

