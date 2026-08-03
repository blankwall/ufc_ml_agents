"""Card-level finish/decision analysis endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.decision_card_service import (
    get_card_analysis,
    get_card_analysis_by_date,
    start_card_analysis,
)

router = APIRouter()


class DecisionFightRequest(BaseModel):
    fighter1: str
    fighter2: str
    finish_odds: Optional[int] = None
    decision_odds: Optional[int] = None


class DecisionCardRequest(BaseModel):
    event_name: Optional[str] = None
    event_date: str
    fights: list[DecisionFightRequest] = Field(min_length=1)
    force: bool = False


@router.post("/decision-cards/analyze")
async def analyze_decision_card(request: DecisionCardRequest):
    return start_card_analysis(
        event_name=request.event_name,
        event_date=request.event_date,
        fights=[fight.model_dump() for fight in request.fights],
        force=request.force,
    )


@router.get("/decision-cards/{card_key}")
async def decision_card_status(card_key: str):
    result = get_card_analysis(card_key)
    if result is None:
        raise HTTPException(status_code=404, detail="Decision card analysis not found.")
    return result


@router.get("/decision-cards")
async def decision_card_by_date(event_date: str):
    result = get_card_analysis_by_date(event_date)
    if result is None:
        raise HTTPException(status_code=404, detail="No cached decision analysis for this date.")
    return result
