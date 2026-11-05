"""동시 요청 비교 측정: 비교 콘솔(demo.py)의 API로 '동시 N건' 실험을 실행하고 결과를 reports/data/burst_compare.json에 저장한다.

실행 전: 다른 터미널에서 python demo.py --port 8201 을 실행해 둔다.
  python scripts/measure_burst.py [모델이름] [사진 수]
"""
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
B = "http://127.0.0.1:8201"
model = sys.argv[1] if len(sys.argv) > 1 else "m3_full_aug_light"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
c = httpx.Client(timeout=300)

cfg = {"model": model, "n": n, "cache": "BOTH", "mode": "burst", "seed": 7, "sample_ids": []}
r = c.post(f"{B}/api/compare", data={"config": json.dumps(cfg), "labels": "[]"})
r.raise_for_status()
while True:
    j = c.get(f"{B}/api/job").json()
    if j["status"] != "running":
        break
    time.sleep(1)
if j["status"] != "done":
    sys.exit(f"실패: {j['status']} {j.get('error')}")
res = j["result"]
(ROOT / "reports/data/burst_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
for run in res["runs"]:
    s = run["summary"]
    sy, a = s["sync"], s["async"]
    print(f"[{run['cache']}] n={s['n']} 정확도 동기 {sy['correct']}/{s['labeled']} 비동기 {a['correct']}/{s['labeled']} | 일치 {s['agreement']['label_match']}/{s['n']}")
    print(f"   동기: 응답 중앙 {sy['http']['median']} 최대 {sy['http']['max']} 전체 {sy['wall_ms']} ms")
    print(f"   비동기: 접수 중앙 {a['accept']['median']} 최대 {a['accept']['max']} 접수완료 {a['accept_wall_ms']} / 최종 중앙 {a['final']['median']} 최대 {a['final']['max']} 전체완료 {a['wall_ms']} ms")
    print(f"   추론시간 중앙 동기 {sy['inference_ms_median']} 비동기 {a['inference_ms_median']} | 캐시HIT 동기 {sy['cache_hits']} 비동기 {a['cache_hits']}")
