import os
from sqlalchemy import JSON
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timezone

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    uploads = relationship(
        "UploadHistory",
        back_populates="owner"
    )


class UploadHistory(Base):
    __tablename__ = "upload_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    fallback_summary = Column(String)
    analysis_data = Column(JSON,nullable=True)
    uploaded_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    owner = relationship(
        "User",
        back_populates="uploads"
    )


Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    try:
        connection = engine.connect()
        print("Connected to the database successfully!")
        connection.close()
    except Exception as e:
        print("Connection failed:", e)