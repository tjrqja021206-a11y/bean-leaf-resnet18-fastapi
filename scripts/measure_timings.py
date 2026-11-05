"""동기/비동기 × 캐시 MISS/HIT 응답시간 측정 (reports/data/timings.csv 생성)
- 같은 이미지 5장을 4개 조건에 모두 사용한다.
- MISS: 요청 직전에 Redis 캐시 DB(1)만 비워 실제 모델 추론을 강제한다.
- HIT : MISS 직후 같은 이미지를 다시 보낸다.
- 비동기 final_ms: 요청 시작 ~ 상태조회가 done이 될 때까지(50ms 간격 폴링 포함).
"""
import csv, json, statistics, subprocess, time
from pathlib import Path
import httpx

ROOT = Path.home() / "fastapi_resnet18"
BASE = "http://127.0.0.1:8000"
OUT = ROOT / "reports/data"

def flush_cache():
    subprocess.run(["docker", "compose", "exec", "-T", "redis", "redis-cli", "-n", "1", "flushdb"],
                   cwd=ROOT, check=True, capture_output=True)

def pick(cls, idx):
    return sorted((ROOT / "dataset/test" / cls).glob("*.jpg"))[idx]

def sync_call(client, path):
    data = path.read_bytes()
    t = time.perf_counter()
    r = client.post(f"{BASE}/api/sync/classify/", files={"image": (path.name, data, "image/jpeg")})
    ms = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    j = r.json()
    return ms, j

def async_call(client, path, poll=0.05):
    data = path.read_bytes()
    t = time.perf_counter()
    r = client.post(f"{BASE}/api/async/classify/", files={"image": (path.name, data, "image/jpeg")})
    http_ms = (time.perf_counter() - t) * 1000
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    while True:
        j = client.get(f"{BASE}/api/async/jobs/{job_id}/").json()
        if j["status"] in ("done", "failed"):
            final_ms = (time.perf_counter() - t) * 1000
            return http_ms, final_ms, j, job_id
        if time.perf_counter() - t > 60:
            raise TimeoutError(job_id)
        time.sleep(poll)

images = [pick("angular_leaf_spot", 30), pick("bean_rust", 30), pick("healthy", 30),
          pick("angular_leaf_spot", 31), pick("bean_rust", 31)]
rows = []
with httpx.Client(timeout=90) as c:
    # 워밍업 (모델 최초 로딩): 비교에서 제외하되 원본 기록은 남긴다
    w = pick("healthy", 40)
    flush_cache(); ms, j = sync_call(c, w)
    rows.append(dict(run_id=0, path="sync", cache_status="WARMUP", http_ms=round(ms,1), final_ms=round(ms,1),
                     inference_ms=j["inference_ms"], job_id="", image=w.name, cache_hit_reported=j["cache_hit"]))
    flush_cache(); h, f, j, jid = async_call(c, w)
    rows.append(dict(run_id=0, path="async", cache_status="WARMUP", http_ms=round(h,1), final_ms=round(f,1),
                     inference_ms=j["inference_ms"], job_id=jid, image=w.name, cache_hit_reported=j["cache_hit"]))
    for i, img in enumerate(images, start=1):
        flush_cache(); ms, j = sync_call(c, img)
        rows.append(dict(run_id=i, path="sync", cache_status="MISS", http_ms=round(ms,1), final_ms=round(ms,1),
                         inference_ms=j["inference_ms"], job_id="", image=img.name, cache_hit_reported=j["cache_hit"]))
        ms, j = sync_call(c, img)  # HIT
        rows.append(dict(run_id=i, path="sync", cache_status="HIT", http_ms=round(ms,1), final_ms=round(ms,1),
                         inference_ms=j["inference_ms"], job_id="", image=img.name, cache_hit_reported=j["cache_hit"]))
        flush_cache(); h, f, j, jid = async_call(c, img)
        rows.append(dict(run_id=i, path="async", cache_status="MISS", http_ms=round(h,1), final_ms=round(f,1),
                         inference_ms=j["inference_ms"], job_id=jid, image=img.name, cache_hit_reported=j["cache_hit"]))
        h, f, j, jid = async_call(c, img)  # HIT
        rows.append(dict(run_id=i, path="async", cache_status="HIT", http_ms=round(h,1), final_ms=round(f,1),
                         inference_ms=j["inference_ms"], job_id=jid, image=img.name, cache_hit_reported=j["cache_hit"]))

with (OUT / "timings.csv").open("w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

summary = {}
for path in ("sync", "async"):
    for cs in ("MISS", "HIT"):
        sel = [r for r in rows if r["path"] == path and r["cache_status"] == cs]
        inf = [r["inference_ms"] for r in sel if r["inference_ms"] is not None]
        summary[f"{path}_{cs}"] = {
            "n": len(sel),
            "http_ms_median": round(statistics.median(r["http_ms"] for r in sel), 1),
            "http_ms_mean": round(statistics.mean(r["http_ms"] for r in sel), 1),
            "final_ms_median": round(statistics.median(r["final_ms"] for r in sel), 1),
            "final_ms_mean": round(statistics.mean(r["final_ms"] for r in sel), 1),
            "inference_ms_median": round(statistics.median(inf), 1) if inf else None,
            "cache_hit_reported": sorted({str(r["cache_hit_reported"]) for r in sel}),
        }
(OUT / "timings_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(open(OUT / "timings.csv", encoding="utf-8").read())
print(json.dumps(summary, ensure_ascii=False, indent=1))
