import hashlib
import io
import time
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, MODEL_PATH
from .hf_model import get_classifier

class InvalidImage(ValueError):
    pass


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    """크기와 디코딩을 검증한 뒤 RGB 이미지를 반환한다."""
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidImage("이미지 파일은 1바이트 이상, 5MB 이하이어야 합니다.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImage("JPG, PNG, WEBP 이미지만 허용합니다.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise InvalidImage("이미지 해상도가 너무 큽니다(최대 1200만 픽셀).")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("올바른 이미지 파일이 아닙니다.") from exc


def classify_image_bytes(image_bytes: bytes) -> dict:
    """모델 추론만 담당한다. Redis 연동은 cache_service.py로 확장한다."""
    image = load_rgb_image(image_bytes)
    started = time.perf_counter()
    raw = get_classifier()(image, top_k=5)
    # Pipeline 버전에 따라 리스트가 한 번 더 감싸질 수 있으므로 정리한다.
    while isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("이미지 분류 모델이 결과를 반환하지 않았습니다.")
    items = [{"label": str(x["label"]), "score": float(x["score"])} for x in raw]
    items.sort(key=lambda x: x["score"], reverse=True)
    return {
        "model_id": MODEL_PATH,
        "top1": items[0],
        "all": items,
        "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        "cache_hit": False,
    }


def cache_key(image_bytes: bytes) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    model = hashlib.sha256(MODEL_PATH.encode("utf-8")).hexdigest()[:16]
    return f"vision:classify:{model}:{digest}"
