#!/usr/bin/env python
"""콩잎 분류 프로젝트 통합 실행 파일: 학습 · 평가 · 비교 · 서비스 적용 · 증강 비교 실험 · GUI

  python pipeline.py gui                       # 동기 vs 비동기 비교 콘솔 GUI 실행 = python demo.py (http://127.0.0.1:8200)
  python pipeline.py train --augment light     # 증강 Fine-tuning (+ --eval 로 test 평가까지)
  python pipeline.py evaluate                  # 아직 평가하지 않은 모델을 test 128장으로 평가
  python pipeline.py compare                   # 저장된 모델 전체 비교표
  python pipeline.py apply m3_full_aug_light  # 서비스(.env + web/worker)에 모델 적용
  python pipeline.py sweep --augment none light medium --seeds 42 43 44   # 증강 효과 반복 비교
  python pipeline.py all --augment light --eval --apply-if-better         # 학습 → 평가 → 비교 → (조건부) 적용
  python pipeline.py status | services start|stop|restart

원칙: 모델 선택은 Validation 점수로 한다. test는 학습·선택에 쓰지 않고 확정 후 보고용으로만 측정한다.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ENV_FILE = ROOT / ".env"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")
SPLIT_DIR = ROOT / "dataset"
HEALTH_URL = "http://127.0.0.1:8000/health"


# ------------------------------------------------------------------ 공통 유틸
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sh(cmd, check=False, timeout=None, capture=True):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip() or f"명령 실패: {cmd}")
    return p


def valid_name(name: str) -> str:
    name = name.strip().replace("artifacts/", "", 1).rstrip("/")
    if not NAME_RE.match(name):
        raise ValueError(f"모델 이름은 영문·숫자·_.- 만 사용할 수 있습니다: {name!r}")
    return name


def model_dir(name: str) -> Path:
    return ARTIFACTS / valid_name(name)


def serving_model() -> str:
    """.env의 MODEL_PATH에서 artifacts/ 아래 모델 이름을 읽는다."""
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"^MODEL_PATH=(.*)$", text, re.M)
    value = (m.group(1).strip() if m else "").rstrip("/")
    return value.replace("artifacts/", "", 1) if value.startswith("artifacts/") else value


# ------------------------------------------------------------------ 모델 목록·지표
def list_models():
    rows = []
    serving = serving_model()
    if not ARTIFACTS.is_dir():
        return rows
    for d in sorted(ARTIFACTS.iterdir()):
        if not (d / "config.json").exists() or not (d / "model.safetensors").exists():
            continue
        report = read_json(d / "training_report.json", {}) or {}
        meta = read_json(d / "run_meta.json", {}) or {}
        settings = report.get("settings") or meta.get("settings") or {}
        test = read_json(d / "test_eval.json")
        rows.append({
            "name": d.name,
            "settings": settings,
            "best_val": report.get("best_val_accuracy"),
            "best_epoch": report.get("best_epoch"),
            "elapsed_sec": report.get("elapsed_sec"),
            "test_accuracy": test.get("accuracy") if test else None,
            "test_correct": test.get("correct") if test else None,
            "test_total": test.get("total") if test else None,
            "test_per_class": test.get("per_class") if test else None,
            "serving": d.name == serving,
            "modified": datetime.fromtimestamp(d.stat().st_mtime).strftime("%m-%d %H:%M"),
            "note": meta.get("note", ""),
        })
    return rows


def _metrics(names, matrix):
    per_class = {}
    for i, name in enumerate(names):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(names))) - tp
        fn = sum(matrix[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": sum(matrix[i])}
    return per_class


def evaluate_models(names, split="test"):
    """모델을 split 이미지로 평가한다. test일 때는 모델 폴더에 test_eval.json으로 캐시한다."""
    sys.path.insert(0, str(ROOT))
    from training.evaluate import evaluate  # torch를 쓰므로 필요할 때만 import

    results = {}
    for name in names:
        d = model_dir(name)
        class_names, matrix, acc, total = evaluate(str(d), SPLIT_DIR, split)
        correct = sum(matrix[i][i] for i in range(len(class_names)))
        result = {"split": split, "accuracy": round(acc, 4), "correct": correct, "total": total,
                  "classes": class_names, "confusion_matrix": matrix,
                  "per_class": _metrics(class_names, matrix), "evaluated_at": now()}
        if split == "test":
            (d / "test_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[name] = result
        print(f"[평가] {name}: {split} accuracy = {acc:.4f} ({correct}/{total})", flush=True)
    return results


# ------------------------------------------------------------------ 학습
def auto_name(args):
    mode = "head" if args.freeze_backbone else "full"
    return f"exp_{args.augment}_{mode}_s{args.seed}_{datetime.now().strftime('%H%M%S')}"


def run_training(name, augment, seed, epochs, batch_size, lr, freeze_backbone, num_workers=4, overwrite=False):
    """training.finetune_aug를 하위 프로세스로 실행한다. 반환: 저장 폴더."""
    out = model_dir(name)
    if out.exists() and (out / "model.safetensors").exists() and not overwrite:
        raise FileExistsError(f"이미 있는 모델입니다: {name} (다른 이름을 쓰거나 --overwrite)")
    cmd = [sys.executable, "-m", "training.finetune_aug", "--data", "dataset", "--output", str(out),
           "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
           "--augment", augment, "--num-workers", str(num_workers)]
    if freeze_backbone:
        cmd.append("--freeze-backbone")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TRANSFORMERS_VERBOSITY": "error", "PYTHONWARNINGS": "ignore"}
    print(f"[학습 시작] {name} | 증강={augment} 시드={seed} 에폭={epochs} 배치={batch_size} lr={lr} "
          f"{'Head-only' if freeze_backbone else 'Full'}", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"학습 실패 (종료코드 {p.returncode})")
    return out


# ------------------------------------------------------------------ 서비스(Docker)
def gather_status():
    info = {"serving_model": serving_model(), "health": False, "containers": [], "gpu": None, "time": now()}
    try:
        import httpx
        info["health"] = httpx.get(HEALTH_URL, timeout=2).status_code == 200
    except Exception:
        pass
    try:
        p = sh(["docker", "compose", "ps", "-a", "--format", "json"], timeout=15)
        for line in p.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                c = json.loads(line)
                info["containers"].append({"name": c.get("Service") or c.get("Name"), "state": c.get("State"),
                                           "status": c.get("Status")})
    except Exception as exc:
        info["docker_error"] = str(exc)[:160]
    try:
        p = sh(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            n, used, total, util = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
            info["gpu"] = {"name": n, "mem_used_mb": int(used), "mem_total_mb": int(total), "util": int(util)}
    except Exception:
        pass
    return info


def services(action):
    cmds = {"start": ["docker", "compose", "up", "-d"], "stop": ["docker", "compose", "stop"],
            "restart": ["docker", "compose", "restart"]}
    if action not in cmds:
        raise ValueError("action은 start / stop / restart 중 하나여야 합니다")
    p = sh(cmds[action], timeout=300)
    print((p.stdout + p.stderr).strip(), flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"docker compose {action} 실패")


def wait_health(limit=120):
    import httpx
    t = time.time()
    while time.time() - t < limit:
        try:
            if httpx.get(HEALTH_URL, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def apply_model(name):
    """.env의 MODEL_PATH를 바꾸고 web·worker를 다시 만든 뒤, 실제 응답의 model_id로 적용을 확인한다."""
    d = model_dir(name)
    if not ((d / "config.json").exists() and (d / "model.safetensors").exists()):
        raise FileNotFoundError(f"모델 파일이 없습니다: {d}")
    new_value = f"artifacts/{d.name}"
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    if ENV_FILE.exists():
        shutil.copy(ENV_FILE, ROOT / ".env.bak")
    if re.search(r"^MODEL_PATH=.*$", text, re.M):
        text = re.sub(r"^MODEL_PATH=.*$", f"MODEL_PATH={new_value}", text, flags=re.M)
    else:
        text += f"\nMODEL_PATH={new_value}\n"
    ENV_FILE.write_text(text, encoding="utf-8")
    print(f"[적용] .env MODEL_PATH={new_value} (이전 값은 .env.bak에 보관)", flush=True)
    sh(["docker", "compose", "up", "-d", "--force-recreate", "web", "worker"], check=True, timeout=300)
    if not wait_health():
        raise RuntimeError("web이 정상 응답하지 않습니다. `docker compose logs web`을 확인하세요.")
    import httpx
    img = next(iter(sorted((SPLIT_DIR / "test").glob("*/*.jpg"))), None)
    if img is None:
        print("[적용] 확인용 test 이미지가 없어 응답 검증은 건너뜀", flush=True)
        return
    r = httpx.post("http://127.0.0.1:8000/api/sync/classify/",
                   files={"image": (img.name, img.read_bytes(), "image/jpeg")}, timeout=120)
    got = r.json().get("model_id") if r.status_code == 200 else None
    if got != new_value:
        raise RuntimeError(f"적용 확인 실패: 응답 model_id={got!r}, 기대={new_value!r}")
    print(f"[적용 완료] 서비스 응답 model_id={got}", flush=True)


# ------------------------------------------------------------------ 출력(표)
def fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}f}"


def print_compare(rows=None):
    rows = rows if rows is not None else list_models()
    print(f"\n{'모델':34} {'증강':7} {'방식':5} {'ep':>3} {'lr':>8} {'시드':>4} {'Val(최고)':>9} {'Test':>14}  서비스")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: -(x["best_val"] or 0)):
        s = r["settings"]
        test = "-" if r["test_accuracy"] is None else f"{r['test_accuracy']:.4f}({r['test_correct']}/{r['test_total']})"
        print(f"{r['name']:34} {str(s.get('augment', '-')):7} {('head' if s.get('freeze_backbone') else 'full') if s else '-':5} "
              f"{str(s.get('epochs', '-')):>3} {str(s.get('lr', '-')):>8} {str(s.get('seed', '-')):>4} "
              f"{fmt(r['best_val']):>9} {test:>14}  {'◀ 서비스 중' if r['serving'] else ''}")
    print("※ 모델 선택은 Val 기준. Test는 확정 후 보고용 (128장: 1장 = 0.78%p)\n")


# ------------------------------------------------------------------ 하위 명령
def cmd_train(args):
    name = valid_name(args.name) if args.name else auto_name(args)
    run_training(name, args.augment, args.seed, args.epochs, args.batch_size, args.lr,
                 args.freeze_backbone, args.num_workers, args.overwrite)
    if args.eval:
        evaluate_models([name])
    print_compare()
    return name


def cmd_evaluate(args):
    names = [valid_name(n) for n in args.models] if args.models else \
        [r["name"] for r in list_models() if r["test_accuracy"] is None or args.force]
    if not names:
        print("평가할 모델이 없습니다 (모두 평가됨). 다시 평가하려면 --force")
        return
    evaluate_models(names, args.split)
    print_compare()


def cmd_sweep(args):
    """증강 강도 × 시드를 반복 학습·평가하고 평균±표준편차로 효과를 비교한다. 가중치는 기본적으로 삭제(보고서·평가만 보존)."""
    runs, tag = [], datetime.now().strftime("%m%d%H%M")
    total = len(args.augment) * len(args.seeds)
    k = 0
    for aug in args.augment:
        for seed in args.seeds:
            k += 1
            name = f"sw{tag}_{aug}_{'head' if args.freeze_backbone else 'full'}_s{seed}"
            print(f"\n===== [{k}/{total}] {name} =====", flush=True)
            run_training(name, aug, seed, args.epochs, args.batch_size, args.lr, args.freeze_backbone, args.num_workers)
            res = evaluate_models([name])[name]
            rep = read_json(model_dir(name) / "training_report.json", {})
            runs.append({"name": name, "augment": aug, "seed": seed, "best_val": rep["best_val_accuracy"],
                         "best_epoch": rep.get("best_epoch"), "test_accuracy": res["accuracy"],
                         "test_correct": res["correct"], "test_total": res["total"], "elapsed_sec": rep.get("elapsed_sec")})
            if not args.keep_weights:  # 가중치는 지우고, 보고서·평가 기록만 reports/data/aug_sweep_runs/로 옮겨 artifacts를 깨끗하게 유지
                (model_dir(name) / "model.safetensors").unlink(missing_ok=True)
                keep_dir = ROOT / "reports" / "data" / "aug_sweep_runs"
                keep_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(model_dir(name)), str(keep_dir / name))
    summary = {}
    for aug in args.augment:
        sel = [r for r in runs if r["augment"] == aug]
        def agg(key):
            v = [r[key] for r in sel]
            return {"mean": round(statistics.mean(v), 4), "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                    "min": round(min(v), 4), "max": round(max(v), 4)}
        summary[aug] = {"n": len(sel), "best_val": agg("best_val"), "test_accuracy": agg("test_accuracy")}
    out = {"created": now(), "settings": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
           "freeze_backbone": args.freeze_backbone, "seeds": args.seeds}, "runs": runs, "summary": summary}
    data_dir = ROOT / "reports" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "aug_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| 증강 | 반복 | Val 평균±표준편차 | Val 범위 | Test 평균±표준편차 | Test 범위 |", "|---|---:|---|---|---|---|"]
    for aug, s in summary.items():
        v, t = s["best_val"], s["test_accuracy"]
        lines.append(f"| {aug} | {s['n']} | {v['mean']:.4f} ± {v['std']:.4f} | {v['min']:.4f}~{v['max']:.4f} | "
                     f"{t['mean']:.4f} ± {t['std']:.4f} | {t['min']:.4f}~{t['max']:.4f} |")
    table = "\n".join(lines)
    (data_dir / "aug_sweep_summary.md").write_text(f"# 증강 비교 실험 ({out['created']})\n\n{table}\n", encoding="utf-8")
    print("\n===== 증강 비교 결과 =====\n" + table + f"\n저장: {data_dir / 'aug_sweep.json'}", flush=True)


def cmd_all(args):
    name = cmd_train(argparse.Namespace(**{**vars(args), "eval": True}))
    if args.apply_if_better:
        rows = {r["name"]: r for r in list_models()}
        new, cur = rows[name], rows.get(serving_model())
        if cur is None or (new["best_val"] or 0) > (cur["best_val"] or 0):
            print(f"[판정] Val {fmt(new['best_val'])} > 현재 서비스 모델 Val {fmt(cur['best_val'] if cur else None)} → 적용")
            apply_model(name)
        else:
            print(f"[판정] Val {fmt(new['best_val'])} ≤ 현재 서비스 모델 {fmt(cur['best_val'])} → 적용하지 않음")


def cmd_status(_):
    s = gather_status()
    print(f"서비스 모델: {s['serving_model'] or '(미지정)'} | 서버 health: {'정상' if s['health'] else '응답 없음'}")
    for c in s["containers"]:
        print(f"  - {c['name']:10} {c['state']:10} {c['status']}")
    print(f"GPU: {s['gpu']}" if s["gpu"] else "GPU: 인식되지 않음")


def cmd_gui(args):
    """동기·비동기 비교 콘솔 GUI(demo.py)를 실행한다."""
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "demo.py"), "--port", str(args.port)]
    import demo
    demo.main()


def add_train_options(p, single_run=True, defaults_epochs=10):
    if single_run:  # sweep은 --augment/--seeds를 여러 개 받으므로 따로 정의한다
        p.add_argument("--augment", choices=("none", "light", "medium"), default="light", help="증강 강도")
        p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=defaults_epochs)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action="store_true", help="Head-only 학습")
    p.add_argument("--num-workers", type=int, default=4)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("train", help="증강 Fine-tuning")
    add_train_options(p)
    p.add_argument("--name", help="모델 이름(기본: 자동)")
    p.add_argument("--eval", action="store_true", help="학습 후 test 평가")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="test 평가(기본: 평가 안 된 모델 전체)")
    p.add_argument("models", nargs="*")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    sub.add_parser("compare", help="모델 비교표").set_defaults(fn=lambda a: print_compare())

    p = sub.add_parser("apply", help="서비스에 모델 적용")
    p.add_argument("model")
    p.set_defaults(fn=lambda a: apply_model(a.model))

    p = sub.add_parser("sweep", help="증강 강도 × 시드 반복 비교 실험")
    add_train_options(p, single_run=False)
    p.add_argument("--augment", nargs="+", choices=("none", "light", "medium"), default=["none", "light", "medium"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--keep-weights", action="store_true", help="가중치 파일도 보존(기본: 삭제)")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("all", help="학습 → test 평가 → 비교 (→ 조건부 적용)")
    add_train_options(p)
    p.add_argument("--name")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--apply-if-better", action="store_true", help="Val이 현재 서비스 모델보다 높을 때만 적용")
    p.set_defaults(fn=cmd_all)

    sub.add_parser("status", help="서비스·GPU 상태").set_defaults(fn=cmd_status)
    p = sub.add_parser("services", help="Docker 서비스 시작/중지/재시작")
    p.add_argument("action", choices=("start", "stop", "restart"))
    p.set_defaults(fn=lambda a: services(a.action))
    p = sub.add_parser("gui", help="동기·비동기 비교 콘솔 GUI 실행 (demo.py)")
    p.add_argument("--port", type=int, default=8200)
    p.set_defaults(fn=cmd_gui)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv if argv is not None else sys.argv[1:] or ["gui"])
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        return 130
    except Exception as exc:  # 사용자에게 읽기 쉬운 오류로 종료
        print(f"[오류] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
