import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

if __name__ == "__main__":
    try:
        connection = engine.connect()
        print("Connected to the database successfully!")
        connection.close()
    except Exception as e:
        print("Connection failed:", e)

from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

class UploadHistory(Base):
    __tablename__ = "upload_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    fallback_summary = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)