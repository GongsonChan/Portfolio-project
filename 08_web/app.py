import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from repository import (
    fetch_backtest_results,
    fetch_basel_capital,
    fetch_latest_final_portfolio,
    fetch_risk_report,
    fetch_var_backtest_summary,
    summarize_backtest,
)
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
    risk = fetch_risk_report(label)
    risk_backtest = fetch_var_backtest_summary(label)
    basel_capital = fetch_basel_capital(label)

    holdings = portfolio_df.to_dict("records")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "label": label,
            "portfolio_date": portfolio_date,
            "holdings": holdings,
            "stats": stats,
            "risk": risk,
            "risk_backtest": risk_backtest,
            "basel_capital": basel_capital,
        },
    )


if __name__ == "__main__":
    import socket

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        # WinError 10014 ("invalid pointer address") shows up intermittently at DIFFERENT WSA
        # socket calls across runs on this machine -- loop self-pipe creation, our own
        # sock.bind()/listen() -- so it's a genuinely non-deterministic low-level flake (not a
        # Winsock LSP; `netsh winsock show catalog` shows only stock mswsock.dll providers),
        # not something localized to one call we can route around for good. It has always
        # succeeded on a retry in testing, so the whole startup sequence (bind, listen, loop
        # creation) is wrapped and retried as one unit instead of chasing the failure call by
        # call. uvicorn (>=0.36) hardcodes ProactorEventLoop on Windows via its own loop_factory
        # regardless of asyncio.set_event_loop_policy(), so SelectorEventLoop is created
        # explicitly here rather than via uvicorn's own startup path.
        loop = None
        last_error = None
        for attempt in range(8):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # SO_REUSEADDR on Windows (unlike POSIX) lets an unrelated process silently
                # bind the SAME address:port even while another process is actively listening
                # on it -- no error, but requests then land unpredictably on whichever process,
                # which is exactly what caused this page to serve stale code from an old
                # process for a while. SO_EXCLUSIVEADDRUSE is the Windows-safe equivalent that
                # actually enforces exclusive ownership.
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind((config.host, config.port))
                sock.listen(2048)
                sock.setblocking(False)
                loop = asyncio.SelectorEventLoop()
                break
            except OSError as e:
                last_error = e
                print(f"[startup] socket/event-loop setup failed (attempt {attempt + 1}/8): {e}", flush=True)
        if loop is None:
            raise RuntimeError("could not start the server after 8 attempts") from last_error

        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve(sockets=[sock]))
        finally:
            loop.close()
    else:
        server.run()
