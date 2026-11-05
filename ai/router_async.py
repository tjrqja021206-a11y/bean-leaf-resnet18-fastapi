from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from .config import MAX_UPLOAD_BYTES, MEDIA_DIR, MODEL_PATH
from .database import SessionLocal
from .models import ImageJob
from .services import InvalidImage, load_rgb_image
from .tasks import classify_job

router = APIRouter(tags=["비동기 큐"])

@router.post("/api/async/classify/", status_code=202)
def enqueue_classify(image: UploadFile = File(...)):
    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="5MB 이하 파일만 허용합니다.")
    try:
        load_rgb_image(image_bytes)  # 큐 등록 전에 유효한 이미지인지 확인
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}.bin"  # 사용자가 보낸 파일명은 사용하지 않는다.
    (MEDIA_DIR / filename).write_bytes(image_bytes)
    with SessionLocal() as db:
        job = ImageJob(image_path=filename, status="queued", model_id=MODEL_PATH)
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
        try:
            classify_job.delay(job_id)  # 추론은 별도 Celery 프로세스에서 수행
        except Exception as exc:
            job.status = "failed"
            job.error_message = "작업 큐 등록 실패: Redis 연결을 확인하세요."
            db.commit()
            raise HTTPException(status_code=503, detail=job.error_message) from exc
    return {"job_id": job_id, "status": "queued", "check_url": f"/api/async/jobs/{job_id}/"}

@router.get("/api/async/jobs/{job_id}/")
def get_job(job_id: int):
    with SessionLocal() as db:
        job = db.get(ImageJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="해당 작업이 없습니다.")
        return {
            "job_id": job.id,
            "status": job.status,
            "model_id": job.model_id,
            "top1": ({"label": job.top1_label, "score": job.top1_score}
                     if job.status == "done" else None),
            "all": job.all_scores if job.status == "done" else None,
            "inference_ms": job.inference_ms,
            "cache_hit": job.cache_hit,
            "error_message": job.error_message if job.status == "failed" else None,
        }
