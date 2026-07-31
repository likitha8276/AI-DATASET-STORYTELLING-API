import pandas as pd
import io
from fastapi import FastAPI , UploadFile
app = FastAPI()
@app.post("/upload")
async def upload_file(file : UploadFile):
    contents = await file.read()
    
    is_csv = file.filename.endswith('csv')
    if is_csv:
     df = pd.read_csv(io.BytesIO(contents))
     return {
        "filename": file.filename,
        "size_bytes": len(contents),
        "is_csv" : is_csv,
        "num_rows": df.shape[0],
        "num_colums" : df.shape[1],
        "num_names": df.columns.tolist(),
        "is_null": df.isnull().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "describe": df.describe().to_dist()
    }
    else :
       return {"error":"Please upload csv file"}

