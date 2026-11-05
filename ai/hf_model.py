from functools import lru_cache

from .config import MODEL_PATH

@lru_cache(maxsize=1)
def get_classifier():
    """서버/워커 프로세스별 최초 요청 때 모델을 로드하고 재사용한다."""
    from transformers import pipeline

    return pipeline("image-classification", model=MODEL_PATH, device=-1)
