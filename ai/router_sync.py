from fastapi import APIRouter, File, HTTPException, UploadFile
from .config import MAX_UPLOAD_BYTES
from .services import InvalidImage
from .cache_service import classify_with_cache

router = APIRouter(tags=["동기 추론"])

@router.post("/api/sync/classify/")
def classify_sync(image: UploadFile = File(...)):
    """HTTP 요청이 추론이 끝날 때까지 기다린다."""
    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="5MB 이하 파일만 허용합니다.")
    try:
        return classify_with_cache(image_bytes)
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
