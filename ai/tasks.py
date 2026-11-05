from .celery_app import celery_app
from .config import MEDIA_DIR, MODEL_PATH
from .database import SessionLocal
from .models import ImageJob
from .cache_service import classify_with_cache

@celery_app.task(name="ai.classify_job")
def classify_job(job_id: int):
    """워커가 DB → 이미지 → Redis/AI → DB 순서로 처리한다."""
    with SessionLocal() as db:
        job = db.get(ImageJob, job_id)
        if job is None:
            return {"error": "job_not_found", "job_id": job_id}
        if job.status == "done":
            return {"job_id": job_id, "status": "done"}
        job.status = "running"
        db.commit()
        image_path = MEDIA_DIR / job.image_path
        try:
            image_bytes = image_path.read_bytes()
            result = classify_with_cache(image_bytes)
            job.model_id = result["model_id"]
            job.top1_label = result["top1"]["label"]
            job.top1_score = result["top1"]["score"]
            job.all_scores = result["all"]
            job.inference_ms = result["inference_ms"]
            job.cache_hit = result["cache_hit"]
            job.status = "done"
            job.error_message = None
            db.commit()
            return {"job_id": job_id, "status": "done"}
        except Exception as exc:
            db.rollback()
            job = db.get(ImageJob, job_id)
            job.status = "failed"
            job.error_message = f"{type(exc).__name__}: {exc}"[:1500]
            db.commit()
            raise
