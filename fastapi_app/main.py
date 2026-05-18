import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers.analyze import router as analyze_router
from routers.backtest import router as backtest_router
from routers.bucket_analysis import router as bucket_analysis_router
from routers.database import router as database_router
from routers.events import router as events_router
from routers.predict import router as predict_router
from routers.scraper import router as scraper_router
from services.the_odds_api_service import run_sync_loop as run_odds_sync_loop, scheduler_enabled as odds_scheduler_enabled
from services.ufcstats_sync_service import run_sync_loop as run_ufcstats_sync_loop, scheduler_enabled as ufcstats_scheduler_enabled

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_tasks = []
    if odds_scheduler_enabled():
        sync_tasks.append(asyncio.create_task(run_odds_sync_loop()))
    if ufcstats_scheduler_enabled():
        sync_tasks.append(asyncio.create_task(run_ufcstats_sync_loop()))
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
app.include_router(predict_router, prefix="/api")
app.include_router(scraper_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/events")


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
    return RedirectResponse(url="/events")


@app.get("/fighter", response_class=HTMLResponse)
async def fighter_page(request: Request):
    return templates.TemplateResponse("fighter.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.ico")
