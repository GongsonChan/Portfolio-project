"""One-command setup: creates .env (prompts for API keys if missing), starts the DB container,
waits for it to be ready, then runs the full pipeline once end-to-end (01 -> 07).

    python setup.py            # quick scope: last ~1 year only (minutes, enough to see it work)
    python setup.py --full     # full historical backfill (slow -- hours, see README.md)

Requires: Docker Desktop running, `pip install -r requirements.txt` already done.
"""
import argparse
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / "01_data_load" / ".env"
ENV_EXAMPLE_PATH = ROOT / "01_data_load" / ".env.example"
PROMPTS = {
    "KRX_API": "KRX_API 키 (data-dbg.krx.co.kr에서 발급)",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY (console.anthropic.com/settings/keys)",
}


def ensure_env() -> None:
    if ENV_PATH.exists():
        print(f"{ENV_PATH} 이미 있음 -- 건너뜀 (키가 비어있으면 직접 채워야 함)")
        return
    print(f"{ENV_PATH} 없음 -- .env.example 기반으로 새로 만듦.\n")
    lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        key = line.split("=", 1)[0]
        if key in PROMPTS and not line.split("=", 1)[1]:
            value = input(f"{PROMPTS[key]}: ").strip()
            out.append(f"{key}={value}")
        else:
            out.append(line)
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n{ENV_PATH} 생성 완료.\n")


def start_db() -> None:
    print("=== Docker DB 컨테이너 기동 ===", flush=True)
    subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT / "01_data_load", check=True)

    sys.path.insert(0, str(ROOT / "01_data_load" / "src"))
    for attempt in range(30):
        try:
            from db import engine  # re-imported each attempt; cached after the first success

            with engine.connect():
                print("DB 준비 완료.\n")
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError("DB가 60초 내에 준비되지 않음 -- Docker Desktop이 켜져 있는지 확인하세요.")


def run(module: str, script: str, *args: str) -> None:
    print(f"\n=== {module}/scripts/{script} {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=ROOT / module, check=True)


def main(full: bool) -> None:
    ensure_env()
    start_db()

    date_args = [] if full else ["--start", (date.today() - timedelta(days=365)).isoformat()]
    if not full:
        print("빠른 세팅 모드: 최근 1년치만 (전체 히스토리는 --full, README.md 참고)\n")

    run("01_data_load", "backfill_krx.py", *date_args)
    run("01_data_load", "backfill_us.py", *date_args)
    run("01_data_load", "backfill_index.py", *date_args)

    run("03_sigma_estimation", "fit_univariate_garch.py")  # 유니버스와 무관, 전체 종목 대상
    run("02_filter", "run_filter.py")  # 유니버스 선정 -- DCC가 이 결과를 가져다 씀
    run("03_sigma_estimation", "fit_dcc.py")

    run("04_views_estimation", "backfill_features.py", *date_args)
    run("04_views_estimation", "run_oof.py", *date_args)
    run("04_views_estimation", "run_sequence_oof.py", *date_args)
    run("04_views_estimation", "run_meta_oof.py")
    run("04_views_estimation", "train_final_models.py")
    run("04_views_estimation", "compute_omega.py")
    run("04_views_estimation", "run_inference.py")

    run("05_black_litterman", "run_black_litterman.py")
    run("06_post_filter", "run_post_filter.py")
    run("07_backtest", "run_backtest.py")

    print("\n=== 설정 완료 ===")
    print("다음부터는 scripts/daily_update.py (평일) / scripts/weekly_update.py (주말)만 돌리면 됩니다.")
    print("대시보드: cd 08_web && python app.py  ->  http://127.0.0.1:8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="전체 히스토리로 백필 (느림, 기본은 최근 1년)")
    args = parser.parse_args()
    main(args.full)
