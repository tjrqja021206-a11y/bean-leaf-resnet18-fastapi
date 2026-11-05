"""통합 테스트: Hugging Face 모델과 Celery 브로커는 스텁으로 바꾼다.
실제 HF 추론/Docker/Redis/Celery 동작을 검증하지는 않는다.
"""
import os
import tempfile
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.mkdtemp()) / "api_test.db")

from fastapi.testclient import TestClient
from PIL import Image
import ai.services as services
import ai.router_async as router_async
import ai.tasks as tasks
from main import app


def image_bytes():
    im = Image.new("RGB", (40, 40), color=(90, 140, 210))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def fake_classifier(image, top_k=5):
    # 00번 Beans 데이터셋의 클래스명을 사용한 가짜 예측 (실제 모델 추론 아님)
    return [{"label": "angular_leaf_spot", "score": 0.92}, {"label": "bean_rust", "score": 0.08}]


def test_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "get_classifier", lambda: fake_classifier)
    monkeypatch.setattr(router_async, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(tasks, "MEDIA_DIR", tmp_path / "jobs")
    # Redis가 실제로 설치되지 않은 테스트에서도 같은 경로를 확인하도록 캐시 호출을 모의 처리한다.
    import ai.router_sync as router_sync
    import ai.cache_service as cache_service
    monkeypatch.setattr(cache_service, "classify_with_cache", services.classify_image_bytes)
    monkeypatch.setattr(router_sync, "classify_with_cache", services.classify_image_bytes)
    monkeypatch.setattr(tasks, "classify_with_cache", services.classify_image_bytes)
    monkeypatch.setattr(router_async.classify_job, "delay", lambda job_id: None)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/vision/sync/").status_code == 200
        assert client.get("/vision/async/").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        valid = client.post("/api/sync/classify/", files={"image": ("a.png", image_bytes(), "image/png")})
        assert valid.status_code == 200
        assert valid.json()["top1"]["label"] == "angular_leaf_spot"
        invalid = client.post("/api/sync/classify/", files={"image": ("x.txt", b"bad", "text/plain")})
        assert invalid.status_code == 400
        oversized = client.post("/api/sync/classify/", files={"image": ("big.jpg", b"a" * (5 * 1024 * 1024 + 1), "image/jpeg")})
        assert oversized.status_code == 413
        queued = client.post("/api/async/classify/", files={"image": ("a.png", image_bytes(), "image/png")})
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        assert client.get(f"/api/async/jobs/{job_id}/").json()["status"] == "queued"
        tasks.classify_job(job_id)  # 테스트에서는 worker 함수를 직접 호출
        done = client.get(f"/api/async/jobs/{job_id}/").json()
        assert done["status"] == "done"
        assert done["top1"]["label"] == "angular_leaf_spot"
        assert client.get("/api/async/jobs/9999999/").status_code == 404


def test_broker_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(router_async, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(router_async.classify_job, "delay", lambda job_id: (_ for _ in ()).throw(ConnectionError("offline")))
    with TestClient(app) as client:
        response = client.post("/api/async/classify/", files={"image": ("a.png", image_bytes(), "image/png")})
        assert response.status_code == 503


def test_cache_hit_avoids_repeated_inference(monkeypatch):
    import json
    import ai.cache_service as cache_service
    store = {}
    class FakeRedis:
        def get(self, key):
            return store.get(key)
        def setex(self, key, seconds, value):
            assert seconds == 600
            store[key] = value
    calls = []
    def fake_inference(data):
        calls.append(data)
        return {"model_id": "test-model", "top1": {"label": "angular_leaf_spot", "score": 1.0},
                "all": [{"label": "angular_leaf_spot", "score": 1.0}], "inference_ms": 8.0, "cache_hit": False}
    monkeypatch.setattr(cache_service, "get_redis", lambda: FakeRedis())
    monkeypatch.setattr(cache_service, "classify_image_bytes", fake_inference)
    first = cache_service.classify_with_cache(b"same image bytes")
    second = cache_service.classify_with_cache(b"same image bytes")
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["inference_ms"] is None
    assert len(calls) == 1


def test_worker_failure_records_failed(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(router_async, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(tasks, "MEDIA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(router_async.classify_job, "delay", lambda job_id: None)
    with TestClient(app) as client:
        queued = client.post("/api/async/classify/", files={"image": ("a.png", image_bytes(), "image/png")})
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        def failed(data):
            raise RuntimeError("fake model crashed")
        monkeypatch.setattr(tasks, "classify_with_cache", failed)
        with pytest.raises(RuntimeError):
            tasks.classify_job(job_id)
        state = client.get(f"/api/async/jobs/{job_id}/").json()
        assert state["status"] == "failed"
        assert "fake model crashed" in state["error_message"]
