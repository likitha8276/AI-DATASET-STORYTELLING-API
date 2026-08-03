import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
num_rows = 9555
num_colums = 14
is_null = {"Data_value": 45, "Suppressed": 9540}
duplicated_rows = 0
summary = "The strongest relationship found was between Period and Data_value, with a correlation of 0.11."

prompt = f"""
You are a data analyst. Write a short, clear 2-3 sentence summary of this dataset for a non-technical reader.

Dataset facts:
- {num_rows} rows, {num_colums} columns
- Missing values: {is_null}
- Duplicate rows: {duplicated_rows}
- Strongest correlation: {summary}

Do not invent any numbers not given above. Only describe what's provided.
"""

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)