#!/usr/bin/env python
"""동기 vs 비동기 비교 콘솔 — 단일 파일 GUI.

내가 고른 사진(업로드 또는 test 사진 선택)을 실제 서비스(Docker: FastAPI web · Celery worker · Redis ·
PostgreSQL)의 동기 API와 비동기 API에 보내서 속도·성능·정확도 차이를 보여준다.
모델은 카드에서 고르며, 각 모델이 BASE(microsoft/resnet-18)에서 무엇이 더해진 모델인지 한눈에 보여준다.

실행:   python demo.py     (또는 ./run.sh)   →  http://127.0.0.1:8200
옵션:   --up       Docker 서비스(redis·postgres·web·worker)를 먼저 시작
        --port N   GUI 포트 (기본 8200)

동작: 모델을 고르면 서비스의 MODEL_PATH를 그 모델로 바꾸고 web·worker를 다시 시작한 뒤 비교한다.
      (.env는 .env.bak으로 백업. 비교 후 원래 모델로 되돌리려면 그 모델을 다시 골라 비교)
주의: 127.0.0.1에만 바인딩되며 인증은 없다(로컬용). 캐시 MISS 측정은 Redis 캐시 DB(1)만 비운다.
"""
import argparse
import io
import json
import math
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

import pipeline as pl  # 서비스 모델 전환(apply_model)·상태 조회 재사용

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "dataset"
SERVICE = "http://127.0.0.1:8000"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")
FILE_RE = re.compile(r"^[\w.\-]+\.(jpg|jpeg|png|webp)$", re.I)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BASE_MODEL = "microsoft/resnet-18"
POLL_SEC = 0.02  # 비동기 상태 조회 간격(최종 완료시간에 포함됨)
SERVICE_MAX_BYTES = 5 * 1024 * 1024  # 서비스 API의 업로드 제한과 같다
MAX_ITEMS = {"sequential": 60, "burst": 40}
CLASSES = sorted(p.name for p in (DATASET / "test").iterdir() if p.is_dir()) if (DATASET / "test").is_dir() else \
    ["angular_leaf_spot", "bean_rust", "healthy"]
AUG_NAME = {"none": "없음", "light": "light", "medium": "medium"}

app = FastAPI(title="동기 vs 비동기 비교 콘솔")


# ------------------------------------------------------------------ 모델 목록 + BASE 대비 변경점
def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def settings_of(d: Path):
    rep = read_json(d / "training_report.json", {}) or {}
    meta = read_json(d / "run_meta.json", {}) or {}
    return rep, meta, (rep.get("settings") or meta.get("settings") or {})


def diff_settings(p: dict, s: dict):
    """부모 모델 대비 달라진 학습 설정을 문장 목록으로 만든다."""
    out = []
    if bool(p.get("freeze_backbone")) != bool(s.get("freeze_backbone")):
        out.append("헤드만 학습 → 전체 가중치 학습" if p.get("freeze_backbone") else "전체 학습 → 헤드만 학습")
    for key, label in (("augment", "증강"), ("epochs", "에폭"), ("batch_size", "배치"), ("lr", "학습률"), ("seed", "시드")):
        a, b = p.get(key), s.get(key)
        if a != b:
            if key == "augment":
                a, b = AUG_NAME.get(a, a), AUG_NAME.get(b, b)
            out.append(f"{label} {a} → {b}")
    return out


def describe(d: Path):
    rep, meta, st = settings_of(d)
    te = read_json(d / "test_eval.json") or {}
    changes = [{"k": "출력층", "v": "ImageNet 1,000클래스 → 콩잎 3클래스(angular_leaf_spot · bean_rust · healthy)로 교체"}]
    if st:
        changes.append({"k": "학습 방식", "v": "분류 헤드만 학습 (Backbone 고정 = Head-only)" if st.get("freeze_backbone")
                        else "전체 가중치 학습 (Full Fine-tuning)"})
        changes.append({"k": "학습 설정", "v": f"{st.get('epochs')}에폭 · 배치 {st.get('batch_size')} · lr {st.get('lr')} · 시드 {st.get('seed')}"})
        changes.append({"k": "데이터 증강", "v": {"none": "없음", "light": "light — 뒤집기 · 회전 15° · 약한 밝기 변화",
                                              "medium": "medium — 확대 · 뒤집기 · 회전 30° · 밝기/채도 변화"}.get(st.get("augment"), str(st.get("augment")))})
    if rep.get("train_count"):
        changes.append({"k": "학습 데이터", "v": f"train {rep['train_count']}장 · val {rep['val_count']}장 (콩잎 3클래스)"})
    if rep.get("best_val_accuracy") is not None:
        at = f"{rep['best_epoch']}에폭 시점" if rep.get("best_epoch") else "Val 최고 시점"
        changes.append({"k": "저장된 가중치", "v": f"{at} (Val 정확도 {rep['best_val_accuracy']:.4f})"})
    vs_parent = None
    parent = meta.get("parent")
    if parent and parent != "BASE" and (ARTIFACTS / parent).is_dir():
        vs_parent = {"parent": parent, "title": (read_json(ARTIFACTS / parent / "run_meta.json", {}) or {}).get("title", parent),
                     "diffs": diff_settings(settings_of(ARTIFACTS / parent)[2], st)}
    return {"name": d.name, "order": meta.get("order", 99), "title": meta.get("title", d.name),
            "short": meta.get("short", ""), "changes": changes, "vs_parent": vs_parent,
            "best_val": rep.get("best_val_accuracy"), "test_accuracy": te.get("accuracy"),
            "test_correct": te.get("correct"), "test_total": te.get("total"), "note": meta.get("note", "")}


def list_models():
    rows, serving = [], pl.serving_model()
    for d in sorted(ARTIFACTS.iterdir()) if ARTIFACTS.is_dir() else []:
        if (d / "config.json").exists() and (d / "model.safetensors").exists():
            m = describe(d)
            m["serving"] = d.name == serving
            rows.append(m)
    return sorted(rows, key=lambda m: (m["order"], m["name"]))


def valid_model(name: str) -> str:
    name = (name or "").strip().replace("artifacts/", "", 1).rstrip("/")
    if not NAME_RE.match(name):
        raise HTTPException(400, "잘못된 모델 이름입니다.")
    d = ARTIFACTS / name
    if not (d / "config.json").exists() or not (d / "model.safetensors").exists():
        raise HTTPException(404, f"모델이 없습니다: {name}")
    return name


# ------------------------------------------------------------------ 사진 선택 (업로드 / test 사진)
def sample_path(sid: str) -> Path:
    """샘플 ID 'test/클래스/파일명' 검증 — test 폴더의 이미지 파일만 허용(경로 조작 차단)."""
    parts = sid.split("/")
    if len(parts) != 3 or parts[0] != "test" or parts[1] not in CLASSES or not FILE_RE.match(parts[2]):
        raise HTTPException(400, "잘못된 샘플 ID입니다.")
    p = DATASET / parts[0] / parts[1] / parts[2]
    if not p.is_file():
        raise HTTPException(404, "이미지가 없습니다.")
    return p


def test_ids(cls=None):
    out = []
    for c in ([cls] if cls else CLASSES):
        d = DATASET / "test" / c
        if d.is_dir():
            out += [f"test/{c}/{p.name}" for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return out


def check_upload(name: str, data: bytes):
    """업로드 사진 검증(서비스 API와 같은 기준: 5MB 이하, 실제 JPEG/PNG/WEBP)."""
    if not data:
        raise HTTPException(400, f"{name}: 빈 파일입니다.")
    if len(data) > SERVICE_MAX_BYTES:
        raise HTTPException(413, f"{name}: 5MB 이하 사진만 비교할 수 있습니다 (서비스 API 제한).")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(400, f"{name}: JPG, PNG, WEBP 사진만 사용할 수 있습니다.")
            im.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, f"{name}: 올바른 이미지 파일이 아닙니다.")


def random_items(n: int, seed: int):
    """클래스 균형을 맞춰 test 사진 n장을 무작위로 고른다(같은 시드 = 같은 사진)."""
    rnd = random.Random(seed)
    per = math.ceil(n / len(CLASSES))
    ids = []
    for c in CLASSES:
        pool = test_ids(c)
        ids += rnd.sample(pool, min(per, len(pool)))
    rnd.shuffle(ids)
    return [item_from_sample(s) for s in ids[:n]]


def item_from_sample(sid: str):
    p = sample_path(sid)
    return {"id": sid, "name": p.name, "truth": sid.split("/")[1], "data": p.read_bytes()}


# ------------------------------------------------------------------ 측정 (실제 서비스 호출)
class Cancelled(Exception):
    pass


JOB = {"status": "idle", "phase": "", "done": 0, "total": 0, "log": [], "result": None, "error": None,
       "cancel": False, "started": None, "ended": None}
_job_lock = threading.Lock()


def jlog(msg):
    JOB["log"] = (JOB["log"] + [f"[{time.strftime('%H:%M:%S')}] {msg}"])[-300:]


def check_cancel():
    if JOB["cancel"]:
        raise Cancelled()


def flush_cache():
    """캐시 MISS를 강제하기 위해 Redis 캐시 DB(1)만 비운다(작업 큐 DB 2·3은 건드리지 않음)."""
    try:
        import redis
        redis.Redis.from_url("redis://127.0.0.1:6379/1", socket_connect_timeout=2).flushdb()
    except Exception:
        subprocess.run(["docker", "compose", "exec", "-T", "redis", "redis-cli", "-n", "1", "flushdb"],
                       cwd=ROOT, capture_output=True)


def sync_call(client, item):
    t = time.perf_counter()
    r = client.post(f"{SERVICE}/api/sync/classify/", files={"image": (item["name"], item["data"], "image/jpeg")})
    ms = (time.perf_counter() - t) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"동기 API 오류 {r.status_code} ({item['name']}): {r.text[:120]}")
    j = r.json()
    return {"ms": round(ms, 1), "label": j["top1"]["label"], "score": round(j["top1"]["score"], 5),
            "all": {x["label"]: round(x["score"], 4) for x in j["all"]},
            "inference_ms": j["inference_ms"], "cache_hit": j["cache_hit"], "model_id": j["model_id"]}


def async_submit(client, item):
    t = time.perf_counter()
    r = client.post(f"{SERVICE}/api/async/classify/", files={"image": (item["name"], item["data"], "image/jpeg")})
    accept = (time.perf_counter() - t) * 1000
    if r.status_code != 202:
        raise RuntimeError(f"비동기 접수 오류 {r.status_code} ({item['name']}): {r.text[:120]}")
    return t, accept, r.json()["job_id"]


def async_poll(client, job_id, t_start, limit=120):
    while True:
        j = client.get(f"{SERVICE}/api/async/jobs/{job_id}/").json()
        if j["status"] in ("done", "failed"):
            return (time.perf_counter() - t_start) * 1000, j
        if time.perf_counter() - t_start > limit:
            raise TimeoutError(f"job {job_id} 시간 초과")
        time.sleep(POLL_SEC)


def async_result(accept, final, j, job_id):
    if j["status"] != "done":
        raise RuntimeError(f"비동기 작업 실패: {j.get('error_message')}")
    return {"accept_ms": round(accept, 1), "final_ms": round(final, 1), "label": j["top1"]["label"],
            "score": round(j["top1"]["score"], 5), "inference_ms": j["inference_ms"], "cache_hit": j["cache_hit"],
            "job_id": job_id, "model_id": j["model_id"]}


def stat(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {"median": round(statistics.median(s), 1), "mean": round(statistics.mean(s), 1),
            "p95": round(s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)], 1), "max": round(s[-1], 1), "min": round(s[0], 1)}


def summarize(rows, wall):
    n = len(rows)
    labeled = [r for r in rows if r["truth"]]
    inf_s = [r["sync"]["inference_ms"] for r in rows if r["sync"]["inference_ms"] is not None]
    inf_a = [r["async"]["inference_ms"] for r in rows if r["async"]["inference_ms"] is not None]
    return {
        "n": n, "labeled": len(labeled),
        "sync": {"correct": sum(r["sync"]["label"] == r["truth"] for r in labeled), "http": stat([r["sync"]["ms"] for r in rows]),
                 "inference_ms_median": stat(inf_s)["median"] if inf_s else None,
                 "cache_hits": sum(bool(r["sync"]["cache_hit"]) for r in rows), "wall_ms": round(wall["sync"], 1)},
        "async": {"correct": sum(r["async"]["label"] == r["truth"] for r in labeled),
                  "accept": stat([r["async"]["accept_ms"] for r in rows]), "final": stat([r["async"]["final_ms"] for r in rows]),
                  "inference_ms_median": stat(inf_a)["median"] if inf_a else None,
                  "cache_hits": sum(bool(r["async"]["cache_hit"]) for r in rows),
                  "wall_ms": round(wall["async_final"], 1), "accept_wall_ms": round(wall["async_accept"], 1)},
        "agreement": {"label_match": sum(r["sync"]["label"] == r["async"]["label"] for r in rows),
                      "max_score_diff": round(max(abs(r["sync"]["score"] - r["async"]["score"]) for r in rows), 5)},
    }


def make_row(i, it, s, a):
    return {"i": i, "id": it["id"], "name": it["name"], "truth": it["truth"], "sync": s, "async": a}


def run_sequential(client, items, cache, base_done):
    rows, w_sync, w_async = [], 0.0, 0.0
    if cache == "HIT":  # 캐시를 미리 채운다(측정 제외)
        flush_cache()
        for it in items:
            sync_call(client, it)
    for i, it in enumerate(items):
        check_cancel()
        if cache == "MISS":
            flush_cache()
        s = sync_call(client, it)
        if cache == "MISS":
            flush_cache()
        t, accept, job_id = async_submit(client, it)
        final, j = async_poll(client, job_id, t)
        a = async_result(accept, final, j, job_id)
        w_sync += s["ms"]
        w_async += a["final_ms"]
        rows.append(make_row(i, it, s, a))
        JOB["done"] = base_done + 2 * (i + 1)
        JOB["phase"] = f"순차 측정 · 캐시 {cache} · {i + 1}/{len(items)}장"
    return rows, summarize(rows, {"sync": w_sync, "async_final": w_async, "async_accept": sum(r["async"]["accept_ms"] for r in rows)})


def run_burst(client, items, cache, base_done):
    n = len(items)
    flush_cache()
    if cache == "HIT":
        for it in items:
            sync_call(client, it)
    check_cancel()
    JOB["phase"] = f"동시 {n}건 · 동기 · 캐시 {cache}"
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        sync_res = [f.result() for f in [ex.submit(sync_call, client, it) for it in items]]
        wall_sync = (time.perf_counter() - t0) * 1000
    JOB["done"] = base_done + n
    check_cancel()
    if cache == "MISS":
        flush_cache()
    JOB["phase"] = f"동시 {n}건 · 비동기 · 캐시 {cache}"
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        subs = [f.result() for f in [ex.submit(async_submit, client, it) for it in items]]
        wall_accept = (time.perf_counter() - t0) * 1000
        JOB["phase"] = f"동시 {n}건 · 비동기 처리 대기 · 캐시 {cache}"
        pending, finals = dict(enumerate(subs)), {}
        while pending:
            check_cancel()
            for k in list(pending):
                t, accept, job_id = pending[k]
                j = client.get(f"{SERVICE}/api/async/jobs/{job_id}/").json()
                if j["status"] in ("done", "failed"):
                    finals[k] = ((time.perf_counter() - t) * 1000, j)
                    del pending[k]
            if time.perf_counter() - t0 > 180:
                raise TimeoutError("비동기 처리 시간 초과")
            time.sleep(POLL_SEC)
        wall_total = (time.perf_counter() - t0) * 1000
    JOB["done"] = base_done + 2 * n
    rows = [make_row(k, it, sync_res[k], async_result(subs[k][1], finals[k][0], finals[k][1], subs[k][2]))
            for k, it in enumerate(items)]
    return rows, summarize(rows, {"sync": wall_sync, "async_final": wall_total, "async_accept": wall_accept})


def run_compare(cfg, items):
    try:
        JOB.update(status="running", phase="준비", done=0, total=0, log=[], result=None, error=None, cancel=False,
                   started=time.strftime("%H:%M:%S"), ended=None)
        if not pl.gather_status()["health"] and not _wait_up(10):
            raise RuntimeError("서비스(8000)가 응답하지 않습니다. 화면 위의 [서비스 시작] 또는 `docker compose up -d`를 실행하세요.")
        model = cfg["model"]
        if pl.serving_model() != model:
            JOB["phase"] = "서비스 모델 전환 (web·worker 재시작)"
            jlog(f"서비스 모델 전환: {pl.serving_model() or '-'} → {model} (약 30초)")
            pl.apply_model(model)
            jlog("전환 완료")
        check_cancel()
        client = httpx.Client(timeout=180, limits=httpx.Limits(max_connections=100))
        conds = ["MISS", "HIT"] if cfg["cache"] == "BOTH" else [cfg["cache"]]
        JOB["total"] = len(conds) * 2 * len(items)
        JOB["phase"] = "워밍업 (모델 첫 로딩은 측정에서 제외)"
        for warm in random_items(3, cfg["seed"] + 999):  # 서로 다른 3장으로 web·worker 모두 예열
            flush_cache()
            sync_call(client, warm)
            flush_cache()
            t, acc, jid = async_submit(client, warm)
            async_poll(client, jid, t)
        jlog(f"워밍업 완료 · 사진 {len(items)}장 · {'직접 선택' if cfg.get('picked') else '무작위(클래스 균형, 시드 ' + str(cfg['seed']) + ')'}")
        runs, done = [], 0
        for cache in conds:
            check_cancel()
            jlog(f"측정 시작: {cfg['mode']} · 캐시 {cache}")
            fn = run_sequential if cfg["mode"] == "sequential" else run_burst
            rows, summ = fn(client, items, cache, done)
            done += 2 * len(items)
            runs.append({"cache": cache, "mode": cfg["mode"], "summary": summ, "rows": rows})
            jlog(f"완료: 캐시 {cache}")
        for r in runs:
            for row in r["rows"]:
                if row["sync"]["model_id"] != f"artifacts/{model}":
                    raise RuntimeError(f"응답의 model_id({row['sync']['model_id']})가 선택한 모델과 다릅니다.")
        mdl = next(m for m in list_models() if m["name"] == model)
        JOB["result"] = {"config": {k: v for k, v in cfg.items() if k != "sample_ids"}, "model": mdl, "runs": runs,
                         "poll_ms": int(POLL_SEC * 1000), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        JOB.update(status="done", phase="완료")
    except Cancelled:
        JOB.update(status="cancelled", phase="중지됨")
    except Exception as exc:
        JOB.update(status="failed", phase="실패", error=f"{type(exc).__name__}: {exc}")
        jlog(f"오류: {exc}")
    finally:
        JOB["ended"] = time.strftime("%H:%M:%S")


def _wait_up(sec):
    t = time.time()
    while time.time() - t < sec:
        try:
            if httpx.get(f"{SERVICE}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


# ------------------------------------------------------------------ API
class CompareIn(BaseModel):
    model: str
    n: int = Field(12, ge=4, le=60)
    cache: Literal["MISS", "HIT", "BOTH"] = "MISS"
    mode: Literal["sequential", "burst"] = "sequential"
    seed: int = Field(7, ge=0, le=99999)
    sample_ids: List[str] = []


@app.get("/api/models")
def api_models():
    return {"models": list_models(), "base": BASE_MODEL, "serving": pl.serving_model(), "classes": CLASSES}


@app.get("/api/status")
def api_status():
    return pl.gather_status()


@app.get("/api/samples")
def api_samples(cls: str = "", page: int = 0, per: int = 18):
    if cls and cls not in CLASSES:
        raise HTTPException(400, "알 수 없는 클래스입니다.")
    ids = test_ids(cls or None)
    per = min(max(per, 6), 60)
    return {"total": len(ids), "page": page, "per": per, "ids": ids[page * per:(page + 1) * per]}


@app.get("/api/image")
def api_image(id: str):
    return FileResponse(sample_path(id))


@app.post("/api/compare")
async def api_compare(config: str = Form(...), labels: str = Form("[]"), images: List[UploadFile] = File(default=[])):
    try:
        b = CompareIn(**json.loads(config))
        label_list = json.loads(labels)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"요청 형식 오류: {exc}")
    model = valid_model(b.model)
    items = []
    for i, f in enumerate(images):
        data = await f.read(SERVICE_MAX_BYTES + 2)
        name = Path(f.filename or f"photo{i + 1}").name
        check_upload(name, data)
        truth = label_list[i] if i < len(label_list) and label_list[i] in CLASSES else None
        items.append({"id": f"upload/{i}/{name}", "name": name, "truth": truth, "data": data})
    items += [item_from_sample(s) for s in b.sample_ids]
    picked = bool(items)
    if not items:
        items = random_items(b.n, b.seed)
    if len(items) > MAX_ITEMS[b.mode]:
        raise HTTPException(400, f"{'동시' if b.mode == 'burst' else '순차'} 실험은 사진 {MAX_ITEMS[b.mode]}장까지 가능합니다 (선택 {len(items)}장).")
    if len(items) < 2:
        raise HTTPException(400, "비교하려면 사진이 2장 이상 필요합니다.")
    with _job_lock:
        if JOB["status"] == "running":
            raise HTTPException(409, "이미 실행 중인 비교가 있습니다.")
        JOB.update(status="running", phase="시작 중", cancel=False, result=None, error=None, log=[], done=0, total=0)
        cfg = {**b.model_dump(), "model": model, "picked": picked, "n": len(items)}
        threading.Thread(target=run_compare, args=(cfg, items), daemon=True).start()
    return {"ok": True, "n": len(items)}


@app.post("/api/services/start")
def api_services_start():
    if JOB["status"] == "running":
        raise HTTPException(409, "비교 실행 중에는 서비스를 제어할 수 없습니다.")
    p = subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise HTTPException(500, (p.stderr or p.stdout)[-300:])
    return {"ok": _wait_up(90)}


@app.get("/api/job")
def api_job():
    return JOB


@app.post("/api/job/cancel")
def api_cancel():
    if JOB["status"] != "running":
        raise HTTPException(409, "실행 중인 비교가 없습니다.")
    JOB["cancel"] = True
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


# ------------------------------------------------------------------ 화면
PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>동기 vs 비동기 비교 콘솔</title>
<style>
:root{--bg:#fff;--ink:#111;--sub:#666;--line:#d4d4d4;--soft:#f4f4f4;--b1:#111;--b2:#8c8c8c;--b3:repeating-linear-gradient(45deg,#333 0 4px,#c8c8c8 4px 8px)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 "Malgun Gothic","Noto Sans KR",system-ui,sans-serif}
header{background:#111;color:#fff;padding:18px 28px}header h1{margin:0;font-size:19px;letter-spacing:-.2px}header p{margin:3px 0 0;color:#bdbdbd;font-size:12.5px}
main{max-width:1040px;margin:0 auto;padding:22px 20px 60px}
section{margin-bottom:26px}
h2{display:flex;align-items:center;gap:9px;margin:0 0 10px;font-size:16px}h2 i{font-style:normal;display:inline-block;width:24px;height:24px;border-radius:50%;background:#111;color:#fff;text-align:center;line-height:24px;font-size:13px}
h2 small{font-weight:400;color:var(--sub);font-size:12.5px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.grow{flex:1}
.hint{color:var(--sub);font-size:12.5px}
button{border:1px solid #111;background:#111;color:#fff;border-radius:6px;padding:8px 18px;font:inherit;font-weight:700;cursor:pointer}
button.sec{background:#fff;color:#111}button.sm{padding:3px 10px;font-size:12.5px}button:disabled{opacity:.35;cursor:not-allowed}button:not(:disabled):hover{opacity:.82}
select,input[type=number]{padding:6px 9px;border:1px solid #999;border-radius:6px;background:#fff;color:#111;font:inherit}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;border:2px solid #111;margin-right:5px;vertical-align:baseline}.dot.on{background:#111}
.st{font-size:12.5px;margin-left:12px;white-space:nowrap}
/* 모델 카드 */
.ladder{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch}.ar{align-self:center;color:#999}
.mc{flex:1;min-width:150px;max-width:230px;border:1px solid var(--line);border-radius:8px;padding:11px 13px;cursor:pointer;background:#fff;position:relative}
.mc:hover{border-color:#111}.mc.sel{background:#111;color:#fff;border-color:#111}.mc.sel .n,.mc.sel .v{color:#cfcfcf}
.mc.base{background:var(--soft);border-style:dashed;cursor:default;max-width:165px}.mc.base:hover{border-color:var(--line)}
.mc .t{font-weight:800;font-size:15px}.mc .s{font-size:12.5px;margin:2px 0}.mc .n{font:11.5px Consolas,monospace;color:var(--sub)}.mc .v{font-size:12px;color:var(--sub);margin-top:4px}
.tag{position:absolute;top:-9px;right:8px;background:#111;color:#fff;border:1px solid #fff;border-radius:20px;padding:0 9px;font-size:11px;font-weight:800}.mc.sel .tag{background:#fff;color:#111;border-color:#111}
details{margin-top:10px}summary{cursor:pointer;font-weight:700;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--sub);font-size:12px;font-weight:700}
td.k{width:110px;color:var(--sub);font-weight:700}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.note{border-left:3px solid #111;background:var(--soft);padding:8px 12px;font-size:12.5px;margin-top:10px}
/* 사진 */
.tabs{display:flex;gap:0;border-bottom:1px solid #111;margin-bottom:10px}.tab{padding:7px 16px;cursor:pointer;color:var(--sub);font-weight:700;border:1px solid transparent;border-bottom:none;margin-bottom:-1px}.tab.on{color:#111;border-color:#111;background:#fff}
.pane{display:none}.pane.on{display:block}
#drop{border:2px dashed #aaa;border-radius:8px;padding:26px 14px;text-align:center;color:var(--sub);cursor:pointer}#drop.hot{border-color:#111;background:var(--soft)}#drop b{color:#111}
.opt{display:flex;gap:6px;flex-wrap:wrap}.opt label{border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:13px}.opt input{margin-right:5px;accent-color:#111}.opt label:has(input:checked){border-color:#111;background:#111;color:#fff}
.tgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;margin-top:8px}
.th{position:relative;border:2px solid transparent;border-radius:8px;overflow:hidden;cursor:pointer;background:var(--soft)}.th img{display:block;width:100%;height:92px;object-fit:cover;filter:grayscale(.15)}
.th.on{border-color:#111}.th.on::after{content:"✓";position:absolute;top:3px;right:3px;background:#111;color:#fff;border-radius:50%;width:19px;height:19px;text-align:center;font-size:12px;line-height:19px;font-weight:800}
.th span{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);color:#fff;font-size:10.5px;padding:1px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.strip{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}.sit{width:104px;font-size:11.5px}.sit .pic{position:relative}.sit img{width:104px;height:80px;object-fit:cover;border-radius:6px;border:1px solid var(--line);display:block}
.sit .x{position:absolute;top:2px;right:2px;background:#111;color:#fff;border-radius:50%;width:18px;height:18px;text-align:center;line-height:17px;cursor:pointer;font-size:11px}
.sit select{padding:2px 3px;font-size:11px;margin-top:3px;width:100%}.sit .tg{color:var(--sub);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cfg{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.cfg label.l{display:block;font-size:12px;color:var(--sub);margin-bottom:4px;font-weight:700}
/* 진행·결과 */
.prog{height:8px;background:var(--soft);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:8px 0}.prog div{height:100%;background:#111;width:0;transition:width .3s}
pre#log{margin:6px 0 0;background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:8px 11px;max-height:120px;overflow:auto;font:12px/1.5 Consolas,monospace;white-space:pre-wrap}
.res{border:1px solid #111;border-radius:10px;padding:14px 16px;margin-bottom:16px}.res h3{margin:0 0 10px;font-size:15px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:14px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:9px 12px}.kpi .a{font-size:11.5px;color:var(--sub);font-weight:700}.kpi .b{font-size:20px;font-weight:800;line-height:1.3}.kpi .c{font-size:12px;color:var(--sub)}
.cmp{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}
.bar{display:grid;grid-template-columns:104px 1fr 74px;gap:8px;align-items:center;margin:6px 0;font-size:12.5px}.track{height:16px;border:1px solid var(--line);border-radius:4px;overflow:hidden}.fill{height:100%}
.read{border-left:3px solid #111;background:var(--soft);padding:9px 14px;margin-top:14px;font-size:13px}.read ul{margin:4px 0 0;padding-left:18px}.read li{margin:3px 0}
.m{font-weight:800}.rimg{width:42px;height:34px;object-fit:cover;border-radius:4px;vertical-align:middle;margin-right:6px;filter:grayscale(.15)}
tr.wr td{background:var(--soft)}
#toast{position:fixed;right:16px;bottom:16px;background:#111;color:#fff;padding:10px 15px;border-radius:8px;display:none;max-width:440px;z-index:9}
</style></head><body>
<header><h1>동기 vs 비동기 비교 콘솔</h1><p>내가 고른 사진을 실제 서비스의 동기 API와 비동기(Celery) API에 보내 속도·성능·정확도 차이를 보여줍니다</p></header>
<main>
<section><div class="row" style="justify-content:space-between"><h2 style="margin:0"><i>1</i>모델 선택 <small>왼쪽 → 오른쪽으로 갈수록 더해진 것</small></h2><div id="stat"></div></div>
<div class="ladder" id="ladder" style="margin-top:14px"></div>
<div id="vsParent" class="hint" style="margin-top:9px"></div>
<details><summary>BASE 대비 변경점 자세히</summary><table id="changes" style="margin-top:6px"></table>
<div class="note" id="foot"></div></details>
<div class="note" id="swWarn" style="display:none"></div></section>

<section><h2><i>2</i>사진 고르기 <small>내 사진 올리기 · test 사진 고르기 (섞어도 됩니다)</small></h2>
<div class="tabs"><div class="tab on" data-t="p1">내 사진 올리기</div><div class="tab" data-t="p2">test 사진에서 고르기</div></div>
<div class="pane on" id="p1"><div id="drop"><b>사진을 끌어다 놓거나 클릭해서 고르세요</b> · Ctrl+V 붙여넣기 · 여러 장<br><span class="hint">JPG / PNG / WEBP, 5MB 이하</span><input id="file" type="file" accept="image/jpeg,image/png,image/webp" multiple hidden></div></div>
<div class="pane" id="p2"><div class="row"><div class="opt" id="clsf"></div><button class="sec sm" id="rnd">무작위 12장 담기</button><button class="sec sm" id="pall">이 페이지 모두 선택</button><span class="hint" id="pinfo"></span><span style="margin-left:auto"><button class="sec sm" id="prev">◀</button> <button class="sec sm" id="next">▶</button></span></div><div class="tgrid" id="tgrid"></div></div>
<div class="row" style="margin-top:14px"><b id="selN">선택한 사진 0장</b><button class="sec sm" id="clr">모두 지우기</button><span class="hint">정답을 아시면 사진 아래에서 지정하세요(정확도 계산용). 아무것도 안 고르면 클래스 균형 무작위 <span id="rN"></span>장을 씁니다.</span></div>
<div class="strip" id="strip"></div></section>

<section><h2><i>3</i>설정 · 실행</h2>
<div class="cfg">
<div><label class="l">캐시 조건</label><div class="opt"><label><input type="radio" name="cache" value="MISS" checked>MISS</label><label><input type="radio" name="cache" value="HIT">HIT</label><label><input type="radio" name="cache" value="BOTH">둘 다</label></div></div>
<div><label class="l">실험 방식</label><div class="opt"><label><input type="radio" name="mode" value="sequential" checked>순차</label><label><input type="radio" name="mode" value="burst">동시</label></div></div>
<div><label class="l">무작위일 때</label><div class="row"><select id="n"><option>8</option><option selected>12</option><option>24</option><option>40</option></select><span class="hint">장 · 시드</span><input id="seed" type="number" value="7" min="0" style="width:70px"></div></div></div>
<div class="row" style="margin-top:14px"><button id="go">비교 실행</button><button id="stop" style="display:none">중지</button><button class="sec" id="svc" style="display:none">서비스 시작</button><span class="hint" id="goHint">MISS = 매번 실제 추론 · HIT = 캐시 재사용 · 동시 = 사진을 한꺼번에 보냄</span></div>
<div id="progCard" style="display:none;margin-top:14px"><div class="row" style="justify-content:space-between"><b id="phase"></b><span class="hint" id="pnum"></span></div><div class="prog"><div id="bar"></div></div><pre id="log"></pre></div></section>

<section id="resCard" style="display:none"><h2><i>4</i>결과</h2><div id="out"></div></section>
</main><div id="toast"></div>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let MODELS=[],SERVING='',SEL='',CLASSES=[],running=false,lastStatus='idle',sel=[],tpage=0,tcls='';
const ms=v=>v==null?'-':(v>=1000?(v/1000).toFixed(2)+' s':Number(v).toFixed(1)+' ms'),pct=v=>(v*100).toFixed(1)+'%',f4=v=>v==null?'-':Number(v).toFixed(4);
function toast(m){const t=$('#toast');t.textContent=m;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',5500)}
async function api(p,o){const r=await fetch(p,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(typeof j.detail==='string'?j.detail:'요청 실패 ('+r.status+')');return j}
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const short=c=>c.replace(/_/g,' '),mark=ok=>ok?'<span class="m">✓</span>':'<span class="m">✗</span>';

// ---------- 1 모델
async function loadModels(keep){const j=await api('/api/models');MODELS=j.models;SERVING=j.serving;CLASSES=j.classes;
 if(!(keep&&MODELS.some(m=>m.name===SEL)))SEL=MODELS.some(m=>m.name===SERVING)?SERVING:(MODELS[0]?.name||'');drawLadder();showModel();drawClsFilter()}
function drawLadder(){$('#ladder').innerHTML=`<div class="mc base"><div class="t">BASE</div><div class="s">microsoft/resnet-18</div><div class="v">ImageNet 사전학습.<br>콩잎 미학습 — 비교 대상 아님</div></div>`+
 MODELS.map(m=>`<div class="ar">→</div><div class="mc ${m.name===SEL?'sel':''}" data-m="${esc(m.name)}">${m.serving?'<span class="tag">서비스 중</span>':''}<div class="t">${esc(m.title)}</div><div class="s">${esc(m.short)}</div><div class="n">${esc(m.name)}</div><div class="v">Val ${f4(m.best_val)}${m.test_accuracy!=null?' · Test '+f4(m.test_accuracy):''}</div></div>`).join('');
 $('#foot').innerHTML='② 는 제공 <code>finetune.py</code>, ③④ 는 <code>finetune_aug.py</code>로 학습했습니다(증강 외 절차는 같고 난수 사용 방식만 다름). ①→② 는 전체 학습 외에 에폭·배치·학습률도 달라서 <b>"전체 학습"의 효과만 따로 볼 수는 없습니다.</b> ③④ 는 ② 와 증강만 다릅니다.'}
function showModel(){const m=MODELS.find(x=>x.name===SEL);if(!m)return;
 $('#changes').innerHTML=m.changes.map(c=>`<tr><td class="k">${esc(c.k)}</td><td>${esc(c.v)}</td></tr>`).join('');
 const v=m.vs_parent;$('#vsParent').innerHTML=`<b>${esc(m.title)}</b> — `+(v?`<b>${esc(v.title)}</b> 대비 달라진 설정: ${v.diffs.length?v.diffs.map(esc).join(' · '):'없음'}`:'BASE에서 출력층만 교체하고 분류 헤드만 학습한 기준 모델');
 const w=$('#swWarn');if(SERVING&&m.name!==SERVING){w.style.display='block';w.textContent=`현재 서비스 모델은 '${SERVING}' 입니다. [비교 실행]을 누르면 서비스를 '${m.name}'로 전환(web·worker 재시작, 약 30초)한 뒤 비교합니다. 되돌리려면 그 모델을 다시 골라 비교하세요.`}else w.style.display='none'}
document.addEventListener('click',ev=>{const c=ev.target.closest('.mc[data-m]');if(c&&!running){SEL=c.dataset.m;drawLadder();showModel()}});
async function loadStatus(){try{const s=await api('/api/status');SERVING=s.serving_model||SERVING;
 $('#stat').innerHTML=`<span class="st"><span class="dot ${s.health?'on':''}"></span>서버 ${s.health?'정상':'꺼짐'}</span>`+(s.containers||[]).map(c=>`<span class="st"><span class="dot ${c.state==='running'?'on':''}"></span>${esc(c.name)}</span>`).join('');
 $('#svc').style.display=s.health||running?'none':'inline-block'}catch(e){}}
$('#svc').onclick=async()=>{$('#svc').disabled=true;toast('서비스를 시작합니다 (최대 1~2분)…');try{await post('/api/services/start');toast('서비스가 시작되었습니다')}catch(e){toast('⚠ '+e.message)}$('#svc').disabled=false;loadStatus()};

// ---------- 2 사진
$$('.tab').forEach(t=>t.onclick=()=>{$$('.tab').forEach(x=>x.classList.toggle('on',x===t));$$('.pane').forEach(p=>p.classList.toggle('on',p.id===t.dataset.t));if(t.dataset.t==='p2')loadTest()});
const drop=$('#drop'),fileIn=$('#file');drop.onclick=()=>fileIn.click();fileIn.onchange=()=>{addFiles([...fileIn.files]);fileIn.value=''};
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>addFiles([...ev.dataTransfer.files]));
document.addEventListener('paste',ev=>{const fs=[...(ev.clipboardData?.files||[])].filter(f=>f.type.startsWith('image/'));if(fs.length)addFiles(fs)});
function addFiles(files){files=files.filter(f=>f.type.startsWith('image/'));if(!files.length)return toast('이미지 파일만 고를 수 있습니다');
 for(const f of files){if(f.size>5*1024*1024){toast(`${f.name}: 5MB를 넘어 제외했습니다`);continue}sel.push({kind:'up',file:f,url:URL.createObjectURL(f),name:f.name,truth:''})}drawStrip()}
async function loadTest(){const j=await api(`/api/samples?cls=${encodeURIComponent(tcls)}&page=${tpage}&per=18`);const last=Math.max(0,Math.ceil(j.total/j.per)-1);
 $('#pinfo').textContent=`${j.total}장 중 ${tpage*j.per+1}~${Math.min(j.total,(tpage+1)*j.per)} (${tpage+1}/${last+1}쪽)`;$('#prev').disabled=tpage<=0;$('#next').disabled=tpage>=last;
 $('#tgrid').innerHTML=j.ids.map(id=>`<div class="th ${sel.some(s=>s.id===id)?'on':''}" data-id="${esc(id)}"><img loading="lazy" src="/api/image?id=${encodeURIComponent(id)}" alt=""><span>${esc(short(id.split('/')[1]))}</span></div>`).join('');$('#tgrid')._ids=j.ids}
function drawClsFilter(){$('#clsf').innerHTML=['',...CLASSES].map(c=>`<label><input type="radio" name="cf" value="${esc(c)}" ${c===tcls?'checked':''}>${c?esc(short(c)):'전체'}</label>`).join('');
 $$('input[name=cf]').forEach(r=>r.onchange=()=>{tcls=r.value;tpage=0;loadTest()})}
$('#prev').onclick=()=>{tpage--;loadTest()};$('#next').onclick=()=>{tpage++;loadTest()};
$('#tgrid').addEventListener('click',ev=>{const t=ev.target.closest('.th');if(!t)return;toggleSample(t.dataset.id);t.classList.toggle('on')});
function toggleSample(id){const i=sel.findIndex(s=>s.id===id);if(i>=0)sel.splice(i,1);else sel.push({kind:'ts',id,url:'/api/image?id='+encodeURIComponent(id),name:id.split('/')[2],truth:id.split('/')[1]});drawStrip()}
$('#pall').onclick=()=>{const ids=$('#tgrid')._ids||[];const allOn=ids.every(id=>sel.some(s=>s.id===id));ids.forEach(id=>{const has=sel.some(s=>s.id===id);if(allOn&&has)toggleSample(id);else if(!allOn&&!has)toggleSample(id)});loadTest()};
$('#rnd').onclick=async()=>{const all=[];for(const c of CLASSES){const r=await api(`/api/samples?cls=${encodeURIComponent(c)}&per=60`);all.push(...r.ids.sort(()=>Math.random()-.5).slice(0,4))}
 all.sort(()=>Math.random()-.5).slice(0,12).forEach(id=>{if(!sel.some(s=>s.id===id))toggleSample(id)});loadTest()};
$('#clr').onclick=()=>{sel=[];drawStrip();if($('#p2').classList.contains('on'))loadTest()};
function drawStrip(){$('#selN').textContent=`선택한 사진 ${sel.length}장`;$('#rN').textContent=$('#n').value;
 $('#strip').innerHTML=sel.map((s,i)=>`<div class="sit"><div class="pic"><img src="${s.url}" alt=""><span class="x" data-rm="${i}">✕</span></div><div class="tg" title="${esc(s.name)}">${esc(s.name)}</div>
 ${s.kind==='up'?`<select data-tr="${i}"><option value="">정답 모름</option>${CLASSES.map(c=>`<option value="${esc(c)}" ${s.truth===c?'selected':''}>${esc(short(c))}</option>`).join('')}</select>`:`<div class="hint">정답: ${esc(short(s.truth))}</div>`}</div>`).join('')}
$('#strip').addEventListener('click',ev=>{const x=ev.target.closest('[data-rm]');if(!x)return;sel.splice(+x.dataset.rm,1);drawStrip();if($('#p2').classList.contains('on'))loadTest()});
$('#strip').addEventListener('change',ev=>{const t=ev.target.closest('[data-tr]');if(t)sel[+t.dataset.tr].truth=t.value});
$('#n').onchange=drawStrip;

// ---------- 3 실행
$('#go').onclick=async()=>{const cache=$('input[name=cache]:checked').value,mode=$('input[name=mode]:checked').value;
 const fd=new FormData();const ups=sel.filter(s=>s.kind==='up');ups.forEach(s=>fd.append('images',s.file,s.name));
 fd.append('labels',JSON.stringify(ups.map(s=>s.truth||'')));
 fd.append('config',JSON.stringify({model:SEL,n:+$('#n').value,cache,mode,seed:+$('#seed').value,sample_ids:sel.filter(s=>s.kind==='ts').map(s=>s.id)}));
 try{await api('/api/compare',{method:'POST',body:fd});window._thumbs=ups.map(s=>s.url);setRunning(true);$('#resCard').style.display='none';$('#progCard').style.display='block';poll()}catch(e){toast('⚠ '+e.message)}};
$('#stop').onclick=async()=>{try{await post('/api/job/cancel')}catch(e){toast('⚠ '+e.message)}};
function setRunning(r){running=r;$('#go').style.display=r?'none':'inline-block';$('#stop').style.display=r?'inline-block':'none'}
async function poll(){try{const j=await api('/api/job');$('#phase').textContent=j.phase||'';$('#pnum').textContent=j.total?`${j.done}/${j.total}`:'';$('#bar').style.width=(j.total?j.done/j.total*100:(j.status==='running'?8:0))+'%';
 $('#log').textContent=(j.log||[]).join('\n');const run=j.status==='running';if(run!==running)setRunning(run);
 if(j.status==='done'&&j.result)render(j.result);if(j.status==='failed')toast('⚠ '+j.error);if(j.status==='cancelled')toast('중지되었습니다');
 if(lastStatus==='running'&&!run){loadModels(true);loadStatus()}lastStatus=j.status;if(run)setTimeout(poll,600)}catch(e){setTimeout(poll,1500)}}

// ---------- 4 결과
function bars(items){const mx=Math.max(...items.map(i=>i.v||0),1);return items.map(i=>`<div class="bar"><span>${i.l}</span><div class="track"><div class="fill" style="width:${Math.max(i.v/mx*100,1.5)}%;background:${i.c}"></div></div><b>${ms(i.v)}</b></div>`).join('')}
function render(r){$('#resCard').style.display='block';const burst=r.config.mode==='burst';
 $('#out').innerHTML=`<div class="hint" style="margin-bottom:10px"><b>${esc(r.model.title)}</b> (${esc(r.model.name)}) · 사진 ${r.config.n}장 (${r.config.picked?'직접 선택':'무작위'}) · ${burst?'동시':'순차'} 실험 · 비동기 최종 시간은 ${r.poll_ms}ms 간격 상태조회 포함 · ${esc(r.finished)}</div>`+r.runs.map(run=>runBlock(run,burst,r)).join('')}
function acc(c,L){return L?`<b>${pct(c/L)}</b> (${c}/${L})`:'<span class="hint">정답 미입력</span>'}
function runBlock(run,burst,r){const S=run.summary,sy=S.sync,as=S.async,n=S.n,L=S.labeled;
 const sMed=sy.http.median,aAcc=as.accept.median,aFin=as.final.median,gain=(sMed-aAcc)/sMed*100,extra=aFin-sMed,same=S.agreement.label_match===n;
 const b1=burst?bars([{l:'동기 전체 완료',v:sy.wall_ms,c:'var(--b1)'},{l:'비동기 전체 접수',v:as.accept_wall_ms,c:'var(--b2)'},{l:'비동기 전체 완료',v:as.wall_ms,c:'var(--b3)'}])
   :bars([{l:'동기 응답',v:sMed,c:'var(--b1)'},{l:'비동기 접수',v:aAcc,c:'var(--b2)'},{l:'비동기 최종 완료',v:aFin,c:'var(--b3)'}]);
 const k1=same?(L&&sy.correct===as.correct?{b:'동기 = 비동기',c:L?`정답 ${sy.correct}/${L} · 라벨 일치 ${S.agreement.label_match}/${n}`:`라벨 일치 ${S.agreement.label_match}/${n}`}:{b:'예측 동일',c:`라벨 일치 ${S.agreement.label_match}/${n}`}):{b:'예측이 다름',c:`라벨 일치 ${S.agreement.label_match}/${n}`};
 const k2=burst?{a:'전체 접수 vs 동기 전체 완료',b:ms(as.accept_wall_ms)+' / '+ms(sy.wall_ms),c:'비동기 접수 | 동기 완료'}:{a:'응답 속도',b:(gain>=0?'−':'+')+Math.abs(gain).toFixed(1)+'%',c:`동기 ${ms(sMed)} → 비동기 접수 ${ms(aAcc)}`};
 const k3=burst?{a:'전체 완료 시간',b:(as.wall_ms-sy.wall_ms>0?'+':'−')+ms(Math.abs(as.wall_ms-sy.wall_ms)),c:`동기 ${ms(sy.wall_ms)} · 비동기 ${ms(as.wall_ms)}`}:{a:'결과를 받기까지',b:(extra>0?'+':'−')+ms(Math.abs(extra)),c:`동기 ${ms(sMed)} · 비동기 ${ms(aFin)}`};
 const lines=[];
 lines.push(same?`<li><b>예측은 동기·비동기가 같습니다</b> (점수 최대 차이 ${S.agreement.max_score_diff}). 같은 모델이 같은 사진을 처리하므로 방식이 정확도를 바꾸지 않습니다.</li>`:`<li><b>동기·비동기 예측이 다른 사진이 있습니다</b> — 같은 모델이므로 예상 밖입니다. 아래 표에서 확인하세요.</li>`);
 if(L)lines.push(`<li>정답을 지정한 ${L}장 기준 — 동기 ${sy.correct}/${L}, 비동기 ${as.correct}/${L}. 장수가 적어 우연의 영향이 큽니다.${r.model.test_accuracy!=null?` 이 모델을 test 128장 전체로 평가한 정확도는 <b>${f4(r.model.test_accuracy)}</b> (${r.model.test_correct}/${r.model.test_total}).`:''}</li>`);
 else lines.push(`<li>정답을 지정하지 않아 정확도는 계산하지 않았습니다.</li>`);
 if(!burst){lines.push(`<li>접수 응답은 동기 응답보다 ${Math.abs(gain).toFixed(1)}% ${gain>=0?'빠르지만':'느리고'}, 결과를 받을 때까지는 ${ms(Math.abs(extra))} ${extra>0?'더 걸렸습니다':'덜 걸렸습니다'}.</li>`);
   if(sy.inference_ms_median!=null&&extra>0)lines.push(`<li>서버 추론은 ${ms(sy.inference_ms_median)}로 짧아서 큐를 거치는 시간과 상태조회 간격이 더 크게 보입니다. 비동기는 추론이 오래 걸릴 때 이득이 큽니다.</li>`)}
 else{const bw=as.wall_ms-sy.wall_ms;lines.push(`<li>${n}건 동시 요청 — 비동기는 전부 접수하는 데 ${ms(as.accept_wall_ms)}, 전체 완료는 동기보다 ${ms(Math.abs(bw))} ${bw>0?'더 걸렸습니다':'덜 걸렸습니다'}. 접수 단계에서도 web이 검증·저장·DB 기록을 하고, worker가 1개라 처리는 순서대로 진행됩니다.</li>`)}
 if(run.cache==='HIT')lines.push(`<li>캐시 HIT: 동기 ${sy.cache_hits}/${n}, 비동기 ${as.cache_hits}/${n}건이 캐시에서 처리되어 추론이 생략되었습니다.</li>`);
 else if(sy.cache_hits||as.cache_hits)lines.push(`<li>MISS 조건이었지만 캐시 HIT가 있었습니다(동기 ${sy.cache_hits}, 비동기 ${as.cache_hits}). 같은 사진을 두 번 골랐을 수 있습니다.</li>`);
 const rows=run.rows.map(x=>{const a=x.sync,b=x.async,t=x.truth,img=x.id.startsWith('test/')?'/api/image?id='+encodeURIComponent(x.id):((window._thumbs&&window._thumbs[x.i])||'');
  const w1=t&&a.label!==t,w2=t&&b.label!==t;
  return `<tr class="${w1||w2?'wr':''}"><td>${img?`<img class="rimg" src="${img}" alt="">`:''}${esc(x.name)}</td><td>${t?esc(short(t)):'<span class="hint">-</span>'}</td><td>${t?mark(!w1)+' ':''}${esc(short(a.label))} ${pct(a.score)}</td><td class="n">${ms(a.ms)}</td><td>${t?mark(!w2)+' ':''}${esc(short(b.label))} ${pct(b.score)}</td><td class="n">${ms(b.accept_ms)}</td><td class="n">${ms(b.final_ms)}</td></tr>`}).join('');
 return `<div class="res"><h3>캐시 ${run.cache==='MISS'?'MISS (매번 실제 추론)':'HIT (캐시 재사용)'} · ${burst?'동시 '+n+'건':'순차 '+n+'장'}</h3>
 <div class="kpis"><div class="kpi"><div class="a">정확도 · 예측</div><div class="b">${k1.b}</div><div class="c">${k1.c}</div></div><div class="kpi"><div class="a">${k2.a}</div><div class="b">${k2.b}</div><div class="c">${k2.c}</div></div><div class="kpi"><div class="a">${k3.a}</div><div class="b">${k3.b}</div><div class="c">${k3.c}</div></div></div>
 <div class="cmp"><div><div class="hint" style="margin-bottom:4px">${burst?'전체 소요 시간':'응답시간 중앙값'}</div>${b1}</div>
 <div><table><tr><th></th><th class="n">동기</th><th class="n">비동기</th></tr>
 <tr><td>정확도</td><td class="n">${acc(sy.correct,L)}</td><td class="n">${acc(as.correct,L)}</td></tr>
 <tr><td>응답 중앙값 / 최대</td><td class="n">${ms(sy.http.median)} / ${ms(sy.http.max)}</td><td class="n">접수 ${ms(as.accept.median)} / ${ms(as.accept.max)}</td></tr>
 <tr><td>결과까지 중앙값 / 최대</td><td class="n">${ms(sy.http.median)} / ${ms(sy.http.max)}</td><td class="n">${ms(as.final.median)} / ${ms(as.final.max)}</td></tr>
 <tr><td>${burst?'전체 소요':'총 소요(합계)'}</td><td class="n">${ms(sy.wall_ms)}</td><td class="n">${ms(as.wall_ms)}</td></tr>
 <tr><td>서버 추론 중앙값</td><td class="n">${ms(sy.inference_ms_median)}</td><td class="n">${ms(as.inference_ms_median)}</td></tr>
 <tr><td>캐시 HIT 수</td><td class="n">${sy.cache_hits}</td><td class="n">${as.cache_hits}</td></tr></table></div></div>
 <div class="read"><b>읽는 법</b><ul>${lines.join('')}</ul></div>
 <details><summary>사진별 결과 (${n}행)</summary><div style="overflow-x:auto"><table><tr><th>사진</th><th>정답</th><th>동기 예측</th><th class="n">동기</th><th>비동기 예측</th><th class="n">접수</th><th class="n">최종</th></tr>${rows}</table></div></details></div>`}
$('#rN').textContent=$('#n').value;loadModels();loadStatus();setInterval(()=>{if(!running)loadStatus()},6000);
api('/api/job').then(j=>{if(j.status==='running'){setRunning(true);$('#progCard').style.display='block';poll()}else if(j.status==='done'&&j.result)render(j.result)}).catch(()=>{});
</script></body></html>"""


# ------------------------------------------------------------------ 실행
def main():
    ap = argparse.ArgumentParser(description="동기 vs 비동기 비교 콘솔 (단일 파일 GUI)")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--up", action="store_true", help="Docker 서비스(redis·postgres·web·worker)를 먼저 시작")
    args = ap.parse_args()
    if args.up:
        print("[demo] docker compose up -d ...", flush=True)
        p = subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT)
        print("[demo] Docker 서비스 시작 " + ("완료" if p.returncode == 0 else "실패(계속 진행)"), flush=True)
        _wait_up(90)
    if not list_models():
        sys.exit("artifacts/ 에 모델이 없습니다. 먼저 학습하세요.")
    health = "정상" if _wait_up(2) else "꺼짐 (./run.sh --up 또는 화면의 [서비스 시작])"
    print(f"[demo] 서비스 모델: {pl.serving_model() or '-'} | 서버(8000): {health}", flush=True)
    print(f"[demo] 열기: http://127.0.0.1:{args.port}   (종료: Ctrl+C)", flush=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
