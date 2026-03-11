from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.backtest_engine import BacktestParams, get_meta, run_backtest

router = APIRouter(tags=["backtest"])


@router.get("/meta")
async def meta():
    """Return available date range, weight classes, and fight count."""
    return get_meta()


@router.post("/backtest")
async def backtest(params: BacktestParams):
    """
    Run a backtest with the supplied parameters.

    Returns summary stats, four Plotly chart JSON blobs, and a fight-level table.
    """
    result = run_backtest(params)
    if "error" in result:
        return JSONResponse(status_code=422, content={"detail": result["error"]})
    return result
