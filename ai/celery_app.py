from celery import Celery
from .config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "vision_ai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["ai.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
