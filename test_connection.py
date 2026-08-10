# test_connection.py
from sqlalchemy import text
from database import engine  # adjust import to your filename

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print("✅ Connected! PostgreSQL version:")
        print(result.fetchone()[0])
except Exception as e:
    print("❌ Connection failed:")
    print(e)