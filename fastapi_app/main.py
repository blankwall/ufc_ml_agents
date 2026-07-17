import asyncio
import json
import re
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers.analyze import router as analyze_router
from routers.backtest import router as backtest_router
from routers.bucket_analysis import router as bucket_analysis_router
from routers.database import router as database_router
from routers.events import router as events_router
from routers.ingest import router as ingest_router
from routers.predict import router as predict_router
from routers.scraper import router as scraper_router
from services.runtime_status import configure_job, get_runtime_health, mark_task_started, mark_task_stopped
from services.health_eval import evaluate_health, list_evidence, resolve_evidence_file, MAX_VIEW_BYTES
from services.sherdog_recovery_service import get_health_status as sherdog_recovery_health_status, recovery_enabled as sherdog_recovery_enabled
from services.the_odds_api_service import get_health_status as odds_health_status
from services.the_odds_api_service import run_sync_loop as run_odds_sync_loop, scheduler_enabled as odds_scheduler_enabled
from services.ufcstats_sync_service import get_health_status as ufcstats_health_status
from services.ufcstats_sync_service import run_sync_loop as run_ufcstats_sync_loop, scheduler_enabled as ufcstats_scheduler_enabled

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"


async def _run_background_loop(job_name: str, runner):
    mark_task_started(job_name)
    try:
        await runner()
    finally:
        mark_task_stopped(job_name, reason="cancelled")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_tasks = []
    configure_job("the_odds_api_sync", enabled=odds_scheduler_enabled())
    configure_job("ufcstats_completed_sync", enabled=ufcstats_scheduler_enabled())
    configure_job("sherdog_recovery", enabled=sherdog_recovery_enabled())
    if odds_scheduler_enabled():
        sync_tasks.append(asyncio.create_task(_run_background_loop("the_odds_api_sync", run_odds_sync_loop)))
    if ufcstats_scheduler_enabled():
        sync_tasks.append(asyncio.create_task(_run_background_loop("ufcstats_completed_sync", run_ufcstats_sync_loop)))
    try:
        yield
    finally:
        for sync_task in sync_tasks:
            sync_task.cancel()
        for sync_task in sync_tasks:
            with suppress(asyncio.CancelledError):
                await sync_task


app = FastAPI(
    title="UFC ML API",
    description="FastAPI-powered UFC model backtesting + event predictions with mar_4_v2",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(analyze_router, prefix="/api")
app.include_router(backtest_router, prefix="/api")
app.include_router(bucket_analysis_router, prefix="/api")
app.include_router(database_router, prefix="/api")
app.include_router(events_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(scraper_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/events")


def _health_payload():
    background_jobs = {
        "the_odds_api_sync": odds_health_status(),
        "sherdog_recovery": sherdog_recovery_health_status(),
        "ufcstats_completed_sync": ufcstats_health_status(),
    }
    verdict = evaluate_health(background_jobs)
    for name, job_eval in verdict["job_eval"].items():
        if name in background_jobs:
            background_jobs[name]["health"] = job_eval
    return {
        "status": verdict["status"],
        "issues": verdict["issues"],
        "data_freshness": verdict["data_freshness"],
        "app": get_runtime_health(),
        "background_jobs": background_jobs,
    }


@app.get("/api/health")
async def health_api():
    return _health_payload()


@app.get("/health", response_class=HTMLResponse)
async def health_page(request: Request):
    payload = _health_payload()
    return templates.TemplateResponse(
        request,
        "health.html",
        {
            "payload": payload,
            "app_health": payload["app"],
            "background_jobs": payload["background_jobs"],
            "issues": payload.get("issues", []),
            "data_freshness": payload.get("data_freshness", {}),
            "evidence": list_evidence(),
        },
    )


@app.get("/health/evidence")
async def health_evidence():
    return {"artifacts": list_evidence()}


@app.get("/health/evidence/{artifact_id}")
async def health_evidence_file(artifact_id: str):
    path = resolve_evidence_file(artifact_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown or missing evidence artifact")
    if path.stat().st_size > MAX_VIEW_BYTES:
        raise HTTPException(status_code=413, detail="Artifact too large to view inline")
    text = path.read_text()
    try:
        return JSONResponse(json.loads(text))
    except json.JSONDecodeError:
        return PlainTextResponse(text)


@app.get("/health.json")
async def health_json():
    return _health_payload()


@app.get("/health/raw")
async def health_raw():
    return {
        **_health_payload(),
    }


@app.get("/events", response_class=HTMLResponse)
async def events_page(request: Request):
    return templates.TemplateResponse("events.html", {"request": request})


@app.get("/bets", response_class=HTMLResponse)
async def bets_page(request: Request):
    return templates.TemplateResponse("bets.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return RedirectResponse(url="/events")


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request):
    return templates.TemplateResponse("ingest.html", {"request": request})


@app.get("/fighter", response_class=HTMLResponse)
async def fighter_page(request: Request):
    return templates.TemplateResponse("fighter.html", {"request": request})


_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _report_title(path: Path) -> str:
    """Best-effort <title> extraction, falling back to a prettified slug."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(4096)
        match = _TITLE_RE.search(head)
        if match:
            return " ".join(match.group(1).split())
    except OSError:
        pass
    return path.stem.replace("_", " ").replace("-", " ").title()


def _list_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for path in sorted(REPORTS_DIR.glob("*.html")):
        reports.append({
            "slug": path.stem,
            "title": _report_title(path),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%b %d, %Y"),
        })
    reports.sort(key=lambda r: r["title"].lower())
    return reports


@app.get("/reports", response_class=HTMLResponse)
async def reports_index(request: Request):
    return templates.TemplateResponse(
        request, "reports.html", {"reports": _list_reports()}
    )


@app.get("/reports/{slug}", response_class=HTMLResponse)
async def report_detail(slug: str):
    if not _SLUG_RE.match(slug):
        raise HTTPException(status_code=404, detail="Report not found")
    path = (REPORTS_DIR / f"{slug}.html").resolve()
    if REPORTS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.ico")
