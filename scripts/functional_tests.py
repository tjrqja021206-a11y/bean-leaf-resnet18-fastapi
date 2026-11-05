"""기능 테스트·오류 시나리오 실행기.
결과: reports/data/functional_results.json, reports/evidence/functional_tests.log
주의: worker와 redis 컨테이너를 잠시 멈췄다가 다시 시작한다.
"""
import datetime
import json
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path.home() / "fastapi_resnet18"
EV = ROOT / "reports/evidence"
DATA = ROOT / "reports/data"
BASE = "http://127.0.0.1:8000"
results, log = [], []


def sh(cmd, timeout=180):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, executable="/bin/bash")
    return (p.stdout + p.stderr).strip()


def dc(args, timeout=180):
    return sh(f"docker compose {args}", timeout)


def L(msg):
    log.append(msg)
    print(msg, flush=True)


def rec(tid, item, procedure, expected, actual, passed):
    results.append(dict(id=tid, item=item, procedure=procedure, expected=expected,
                        actual=actual, verdict="PASS" if passed else "FAIL"))
    L(f"[{tid}] {'PASS' if passed else 'FAIL'} | {item} | 기대={expected} | 실제={actual}")


def img(cls, i):
    return sorted((ROOT / "dataset/test" / cls).glob("*.jpg"))[i]


def post(c, url, path, name=None, ctype="image/jpeg", data=None):
    data = path.read_bytes() if data is None else data
    return c.post(BASE + url, files={"image": (name or (path.name if path else "x"), data, ctype)})


def wait_job(c, job_id, limit=90):
    t = time.time()
    while time.time() - t < limit:
        j = c.get(f"{BASE}/api/async/jobs/{job_id}/").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    return c.get(f"{BASE}/api/async/jobs/{job_id}/").json()


def flush_cache():
    dc("exec -T redis redis-cli -n 1 flushdb")


L(f"# 기능 테스트 실행 시각: {datetime.datetime.now().isoformat(timespec='seconds')}")
with httpx.Client(timeout=120) as c:
    r = c.get(BASE + "/health")
    rec("T-01", "FastAPI 헬스", "GET /health", "200 / ok", f"{r.status_code} / {r.json()}",
        r.status_code == 200 and r.json() == {"status": "ok"})

    for tid, cls in (("T-02a", "angular_leaf_spot"), ("T-02b", "bean_rust"), ("T-02c", "healthy")):
        p = img(cls, 5)
        r = post(c, "/api/sync/classify/", p)
        j = r.json()
        rec(tid, f"동기 추론 ({cls})", f"POST /api/sync/classify/ ({p.name})", f"200 / 정답={cls}",
            f"{r.status_code} / {j['top1']['label']} ({j['top1']['score']:.3f})",
            r.status_code == 200 and j["top1"]["label"] == cls)

    p = img("bean_rust", 6)
    flush_cache()
    r = post(c, "/api/async/classify/", p)
    j = r.json()
    rec("T-03", "비동기 접수", f"POST /api/async/classify/ ({p.name})", "202 / job_id / queued / check_url",
        f"{r.status_code} / job_id={j.get('job_id')} / {j.get('status')} / {j.get('check_url')}",
        r.status_code == 202 and "job_id" in j)
    job_id = j["job_id"]
    res = wait_job(c, job_id)
    rec("T-04", "비동기 완료", f"GET /api/async/jobs/{job_id}/ (폴링)", "200 / done / 정답=bean_rust",
        f"status={res['status']} / {res['top1']['label'] if res.get('top1') else None} / cache_hit={res.get('cache_hit')}",
        res["status"] == "done" and res["top1"]["label"] == "bean_rust")
    (EV / "api_async_job_done.json").write_text(
        json.dumps({"accepted": j, "final": res}, ensure_ascii=False, indent=2), encoding="utf-8")

    out = dc("exec -T redis redis-cli ping")
    rec("T-05", "Redis 연결", "docker compose exec redis redis-cli ping", "PONG", out, out == "PONG")

    out = dc("exec -T postgres psql -U vision_user -d vision_db -c "
             f"\"SELECT id,status,model_id,top1_label,round(top1_score::numeric,4) AS score,cache_hit,inference_ms "
             f"FROM image_jobs WHERE id={job_id};\"")
    (EV / "db_job_row.txt").write_text(out + "\n", encoding="utf-8")
    line = out.splitlines()[2].strip() if len(out.splitlines()) > 2 else out
    rec("T-06", "DB 기록", f"image_jobs id={job_id} 조회", f"id={job_id}, done", line,
        f"{job_id} | done" in out)

    r = post(c, "/api/sync/classify/", None, "x.txt", "text/plain", b"this is not an image")
    r2 = post(c, "/api/async/classify/", None, "x.txt", "text/plain", b"this is not an image")
    rec("T-07", "불량 입력(.txt)", "sync·async에 텍스트 파일 업로드", "400",
        f"sync {r.status_code} {r.json()} / async {r2.status_code}",
        r.status_code == 400 and r2.status_code == 400)

    r = c.get(f"{BASE}/api/async/jobs/999999/")
    rec("T-08", "없는 작업번호", "GET /api/async/jobs/999999/", "404", f"{r.status_code} {r.json()}",
        r.status_code == 404)

    p = img("healthy", 7)
    flush_cache()
    a = post(c, "/api/sync/classify/", p).json()
    b = post(c, "/api/sync/classify/", p).json()
    rec("T-09", "캐시 반복", f"동일 파일 2회 ({p.name}), 캐시 비운 뒤", "MISS → HIT",
        f"cache_hit: {a['cache_hit']} → {b['cache_hit']} / inference_ms: {a['inference_ms']} → {b['inference_ms']}",
        a["cache_hit"] is False and b["cache_hit"] is True)

    big = b"a" * (5 * 1024 * 1024 + 1)
    r = post(c, "/api/sync/classify/", None, "big.jpg", "image/jpeg", big)
    r2 = post(c, "/api/async/classify/", None, "big.jpg", "image/jpeg", big)
    rec("T-10", "용량 초과(5MB+1)", "sync·async에 5MB+1바이트 업로드", "413",
        f"sync {r.status_code} / async {r2.status_code}", r.status_code == 413 and r2.status_code == 413)

    # --- Worker 정지 ---
    L("... worker 정지 시험")
    dc("stop worker")
    time.sleep(2)
    p = img("angular_leaf_spot", 8)
    flush_cache()
    r = post(c, "/api/async/classify/", p)
    jid = r.json()["job_id"]
    time.sleep(5)
    st = c.get(f"{BASE}/api/async/jobs/{jid}/").json()
    row = dc(f"exec -T postgres psql -U vision_user -d vision_db -tA -c \"SELECT id,status FROM image_jobs WHERE id={jid};\"")
    rec("T-11", "Worker 미실행", "worker 정지 후 비동기 접수, 5초 대기", "접수 202, 상태 queued 유지",
        f"{r.status_code}, 5초 뒤 상태={st['status']}, DB={row}",
        r.status_code == 202 and st["status"] == "queued")
    dc("start worker")
    fin = wait_job(c, jid, 90)
    rec("T-11b", "Worker 재시작 후 처리", f"같은 job_id={jid} 재조회", "done(관찰 결과를 그대로 기록)",
        f"status={fin['status']} / {fin['top1']['label'] if fin.get('top1') else None}", fin["status"] == "done")

    # --- Redis 정지 ---
    L("... redis 정지 시험")
    dc("stop redis")
    time.sleep(2)
    p = img("healthy", 9)
    t = time.time()
    r = post(c, "/api/async/classify/", p)
    el = time.time() - t
    detail = r.json().get("detail")
    row = dc("exec -T postgres psql -U vision_user -d vision_db -tA -c "
             "\"SELECT id,status,error_message FROM image_jobs ORDER BY id DESC LIMIT 1;\"")
    rec("T-12", "Redis 장애: 비동기 접수", "redis 정지 후 POST /api/async/classify/", "503 + failed 행 기록",
        f"{r.status_code} ({el:.1f}s) {detail} / DB: {row}", r.status_code == 503 and "failed" in row)
    p2 = img("bean_rust", 9)
    r = post(c, "/api/sync/classify/", p2)
    j = r.json()
    rec("T-13", "Redis 장애: 동기 폴백", "redis 정지 상태에서 POST /api/sync/classify/", "200 (캐시 없이 추론)",
        f"{r.status_code} / {j['top1']['label']} / cache_hit={j['cache_hit']}",
        r.status_code == 200 and j["cache_hit"] is False)
    dc("start redis")
    for _ in range(30):
        if "healthy" in dc("ps redis --format '{{.Status}}'"):
            break
        time.sleep(2)
    time.sleep(3)
    p3 = img("angular_leaf_spot", 10)
    flush_cache()
    r = post(c, "/api/async/classify/", p3)
    fin = wait_job(c, r.json()["job_id"], 60) if r.status_code == 202 else {"status": f"enqueue {r.status_code}"}
    rec("T-14", "Redis 복구 후 비동기", "redis 재시작 후 비동기 접수·완료", "202 → done",
        f"{r.status_code} → {fin['status']}", r.status_code == 202 and fin["status"] == "done")
    if fin["status"] != "done":
        L("... 복구 실패 → worker 재시작 후 재시도")
        dc("restart worker")
        time.sleep(8)
        r = post(c, "/api/async/classify/", p3)
        fin = wait_job(c, r.json()["job_id"], 60)
        L(f"재시도 결과: {r.status_code} → {fin['status']}")

    for tid, url in (("T-15a", "/vision/sync/"), ("T-15b", "/vision/async/"),
                     ("T-15c", "/docs"), ("T-15d", "/static/style.css")):
        r = c.get(BASE + url)
        rec(tid, f"화면/정적 파일 {url}", f"GET {url}", "200", str(r.status_code), r.status_code == 200)

local = sh(".venv/bin/python -m pytest -q 2>&1 | tail -1")
cont = dc("exec -T web pytest -q tests/test_api.py 2>&1 | tail -1")
rec("T-16", "모의(mock) 테스트 — 로컬", "python -m pytest -q", "4 passed", local, "4 passed" in local)
rec("T-17", "모의(mock) 테스트 — 컨테이너", "docker compose exec web pytest -q tests/test_api.py",
    "4 passed", cont, "4 passed" in cont)

(EV / "functional_tests.log").write_text("\n".join(log) + "\n", encoding="utf-8")
(DATA / "functional_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nPASS", sum(r["verdict"] == "PASS" for r in results), "/", len(results))
