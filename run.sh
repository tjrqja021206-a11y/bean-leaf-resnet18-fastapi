#!/usr/bin/env bash
# 통합 실행 파일
#   ./run.sh                          → 동기 vs 비동기 비교 콘솔 GUI (python demo.py, http://127.0.0.1:8200)
#   ./run.sh --up                     → Docker 서비스(redis·postgres·web·worker)도 함께 시작
#   ./run.sh train --augment light --eval   → pipeline.py 명령 (학습·평가·비교·적용 등)
cd "$(dirname "$0")"
source .venv/bin/activate
if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
  exec python demo.py "$@"
fi
exec python pipeline.py "$@"
