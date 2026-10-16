import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from repository import fetch_backtest_results, fetch_latest_final_portfolio, summarize_backtest
from scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Scheduling lives inside the app process itself -- as long as this server is running
    # (wherever it's deployed), daily/weekly updates happen with it. No OS cron/Task Scheduler
    # setup needed (see scheduler.py for the times).
    scheduler = start_scheduler()
    yield
    scheduler.shutdown()


app = FastAPI(title="Small Funds", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, label: str = "default"):
    portfolio_df, portfolio_date = fetch_latest_final_portfolio(label)
    backtest_df = fetch_backtest_results(label)
    stats = summarize_backtest(backtest_df)

    holdings = portfolio_df.to_dict("records")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "label": label,
            "portfolio_date": portfolio_date,
            "holdings": holdings,
            "stats": stats,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
