"""Card-level finish/decision analysis endpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.decision_card_service import (
    get_card_analysis,
    get_card_analysis_by_date,
    start_card_analysis,
)

router = APIRouter()
DecisionCardView = Literal["all", "signals", "actionable"]


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


def _apply_view(result: dict, view: DecisionCardView) -> dict:
    payload = deepcopy(result)
    fights = payload.get("fights", [])
    summary = {
        "total": len(fights),
        "signals": sum(1 for fight in fights if fight.get("result", {}).get("eligible") is True),
        "strong": sum(1 for fight in fights if fight.get("result", {}).get("tier") == "strong"),
        "actionable": sum(1 for fight in fights if fight.get("result", {}).get("bet") is True),
        "errors": sum(1 for fight in fights if fight.get("result", {}).get("bet") == "error"),
    }
    if view == "signals":
        fights = [
            fight for fight in fights
            if fight.get("result", {}).get("eligible") is True
        ]
    elif view == "actionable":
        fights = [
            fight for fight in fights
            if fight.get("result", {}).get("bet") is True
        ]
    payload["view"] = view
    payload["summary"] = summary
    payload["returned_fights"] = len(fights)
    payload["fights"] = fights
    return payload


@router.post("/decision-cards/analyze")
async def analyze_decision_card(request: DecisionCardRequest):
    return start_card_analysis(
        event_name=request.event_name,
        event_date=request.event_date,
        fights=[fight.model_dump() for fight in request.fights],
        force=request.force,
    )


@router.get("/decision-cards/{card_key}")
async def decision_card_status(card_key: str, view: DecisionCardView = "all"):
    result = get_card_analysis(card_key)
    if result is None:
        raise HTTPException(status_code=404, detail="Decision card analysis not found.")
    return _apply_view(result, view)


@router.get("/decision-cards")
async def decision_card_by_date(event_date: str, view: DecisionCardView = "all"):
    result = get_card_analysis_by_date(event_date)
    if result is None:
        raise HTTPException(status_code=404, detail="No cached decision analysis for this date.")
    return _apply_view(result, view)
