from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base


DATABASE_URL = "sqlite:///./data/eduagentx.db"
Path("data").mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables used by the MVP backend."""
    from backend.database import models  # noqa: F401 - registers ORM metadata

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields one SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
