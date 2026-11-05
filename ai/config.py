import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/resnet-18")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://vision_user:vision_pass@127.0.0.1:5432/vision_db",
)
REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://127.0.0.1:6379/1")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/2")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/3")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MEDIA_DIR = BASE_DIR / "media" / "jobs"
