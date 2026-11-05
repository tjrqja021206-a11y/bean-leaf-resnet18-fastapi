from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base

def utcnow():
    return datetime.now(timezone.utc)

class ImageJob(Base):
    __tablename__ = "image_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(150), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    model_id: Mapped[str] = mapped_column(String(255), default="")
    top1_label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    top1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    all_scores: Mapped[list | None] = mapped_column(JSON, nullable=True)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cache_hit: Mapped[bool | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
