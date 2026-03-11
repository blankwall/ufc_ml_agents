from fastapi import APIRouter
from services.predict_service import get_events_data

router = APIRouter()


@router.get("/events")
async def api_events():
    """Return all events with fight predictions and outcome results."""
    return get_events_data()
