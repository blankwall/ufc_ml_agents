from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers.analyze import router as analyze_router
from routers.backtest import router as backtest_router
from routers.events import router as events_router
from routers.predict import router as predict_router
from routers.scraper import router as scraper_router

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="UFC ML API",
    description="FastAPI-powered UFC model backtesting + event predictions with mar_4_v2",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(analyze_router, prefix="/api")
app.include_router(backtest_router, prefix="/api")
app.include_router(events_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(scraper_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/events")


@app.get("/events", response_class=HTMLResponse)
async def events_page(request: Request):
    return templates.TemplateResponse("events.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.ico")
