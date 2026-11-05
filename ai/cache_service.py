import json
import logging
from functools import lru_cache
from .config import REDIS_CACHE_URL
from .services import cache_key, classify_image_bytes

log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_redis():
    from redis import Redis
    return Redis.from_url(REDIS_CACHE_URL, decode_responses=True,
                          socket_connect_timeout=1, socket_timeout=1)


def classify_with_cache(image_bytes: bytes) -> dict:
    """Redis 장애 시에도 모델 추론을 진행하는 실습용 캐시 폴백."""
    key = cache_key(image_bytes)
    try:
        saved = get_redis().get(key)
        if saved:
            result = json.loads(saved)
            return {**result, "cache_hit": True, "inference_ms": None}
    except Exception as exc:
        log.warning("Redis 조회 불가; 모델 추론으로 전환: %s", exc)

    result = classify_image_bytes(image_bytes)
    try:
        get_redis().setex(key, 600, json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        log.warning("Redis 캐시 저장 실패: %s", exc)
    return result
