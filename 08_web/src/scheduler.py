import subprocess
import sys
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

ROOT = Path(__file__).resolve().parent.parent.parent  # project root (one level above 08_web)

# Runs inside the web app process itself -- as long as `python app.py` (or uvicorn) is running,
# daily/weekly updates happen with it, no OS-level cron/Task Scheduler needed. Times are local
# server time; adjust if deployed somewhere with a different timezone than intended.
DAILY_HOUR, DAILY_MINUTE = 8, 0  # weekdays
WEEKLY_HOUR, WEEKLY_MINUTE = 9, 0  # Saturday


def _run(script: str) -> None:
    log_path = ROOT / "scripts" / f"{script.removesuffix('.py')}.log"
    print(f"[scheduler] {script} 시작", flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        subprocess.run([sys.executable, f"scripts/{script}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"[scheduler] {script} 완료 (로그: {log_path})", flush=True)


def start_scheduler() -> BackgroundScheduler:
    """BackgroundScheduler runs jobs in their own worker thread, so a long daily/weekly run
    (tens of minutes to hours -- see DAILY_UPDATE.md) doesn't block the web server itself."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: _run("daily_update.py"),
        CronTrigger(day_of_week="mon-fri", hour=DAILY_HOUR, minute=DAILY_MINUTE),
        id="daily_update",
    )
    scheduler.add_job(
        lambda: _run("weekly_update.py"),
        CronTrigger(day_of_week="sat", hour=WEEKLY_HOUR, minute=WEEKLY_MINUTE),
        id="weekly_update",
    )
    scheduler.start()
    print(
        f"[scheduler] 시작됨 -- daily_update.py 평일 {DAILY_HOUR:02d}:{DAILY_MINUTE:02d}, "
        f"weekly_update.py 토요일 {WEEKLY_HOUR:02d}:{WEEKLY_MINUTE:02d}",
        flush=True,
    )
    return scheduler
