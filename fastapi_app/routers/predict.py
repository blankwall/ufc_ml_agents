"""
POST /api/predict
-----------------
One-off fight prediction given two fighter names, an optional fight date,
and optional American odds.  No caching — runs fresh on every call.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.schema import Fighter
from services.bet_evaluator import SKIP_REASONS, evaluate_bet_decision
from services.historical_context_service import describe_historical_context
from services.predict_service import (
    FIGHTER_ALIASES,
    MatchupFeatureExtractor,
    _fight_count_as_of,
    _is_wmma,
    _resolve_fighter,
    _score_row,
)

router = APIRouter()

_DB_PATH     = ROOT_DIR / "data" / "ufc_database.db"
_CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"
_engine  = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)

# ── request / response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    fighter1:      str
    fighter2:      str
    fight_date:    Optional[date] = None   # used as as_of_date for feature extraction
    fighter1_odds: Optional[int]  = None   # American odds, e.g. -380 or +310
    fighter2_odds: Optional[int]  = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _load_betting_filters() -> dict:
    """Read filter thresholds + WMMA rules from config/betting_config.json."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        return {
            "filters": cfg.get("filters", {}),
            "wmma":    cfg.get("wmma_rules", {}),
        }
    except Exception:
        return {}


def _evaluate_bet(
    *,
    fighter1_name: str = "",
    fighter2_name: str = "",
    pick_slot: str = "fighter1",
    pick_model_prob: float,   # 0–1 — model conviction on the picked side
    pick_mkt_prob:   float,   # 0–1 — implied market prob on the picked side
    pick_odds:       Optional[int],
    is_favorite:     bool,
    is_wmma:         Optional[bool],
    f1_count:        int,
    f2_count:        int,
    as_of_date:      Optional[datetime] = None,
) -> dict:
    """Apply current betting_config rules and return (bet, skip_code, reason)."""
    cfg          = _load_betting_filters()
    filters      = cfg.get("filters", {})
    wmma_rules   = cfg.get("wmma", {})
    return evaluate_bet_decision(
        fighter1_name=fighter1_name,
        fighter2_name=fighter2_name,
        pick_slot=pick_slot,
        pick_model_prob=pick_model_prob,
        pick_mkt_prob=pick_mkt_prob,
        pick_odds=pick_odds,
        is_favorite=is_favorite,
        is_wmma=is_wmma,
        f1_count=f1_count,
        f2_count=f2_count,
        filters=filters,
        wmma_rules=wmma_rules,
        as_of_date=as_of_date,
    )


def _matchup_wmma_flag(session, fighter1_id: int, fighter2_id: int) -> Optional[bool]:
    w1 = _is_wmma(session, fighter1_id)
    w2 = _is_wmma(session, fighter2_id)
    if w1 is True or w2 is True:
        return True
    if w1 is None and w2 is None:
        return None
    return False


def _predict_review_fields(bet_eval: dict) -> dict:
    if bet_eval.get("decision_source") != "golden_elo_reopen":
        return {
            "review_bucket": None,
            "review_tier": None,
            "review_label": None,
        }
    return {
        "review_bucket": bet_eval.get("review_bucket"),
        "review_tier": bet_eval.get("review_tier"),
        "review_label": bet_eval.get("review_label"),
    }


def _predict_decision_label(bet_eval: dict) -> str:
    source = bet_eval.get("decision_source")
    if source == "golden_elo_reopen":
        return "Bet (Golden ELO)"
    if bet_eval.get("bet"):
        return "Bet"
    return "Pass"


def _predict_explanation(bet_eval: dict) -> str | None:
    skip_code = bet_eval.get("skip_code")
    review_label = bet_eval.get("review_label")
    source = bet_eval.get("decision_source")

    if source == "golden_elo_reopen" and review_label:
        return f"Golden ELO reopen: {review_label}"

    if skip_code == "F1":
        return "Pass: the favorite does not clear the confidence threshold."
    if skip_code == "F2":
        return "Pass: the favorite price is too expensive."
    if skip_code == "F3":
        return "Pass: the favorite edge is too small."
    if skip_code == "U1":
        return "Pass: the underdog does not clear the confidence threshold."
    if skip_code == "U2":
        return "Pass: the underdog edge is too small."
    if skip_code == "U3":
        return "Pass: the underdog price is outside the allowed range."
    if skip_code == "W1":
        return "Pass: WMMA requires a larger edge."
    if skip_code == "D1":
        return "Pass: there is not enough historical fight data."
    if skip_code == "ERR":
        return "Pass: prediction failed."

    if bet_eval.get("bet"):
        return "Bet: the model clears the current betting rules."
    return None


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_fight(req: PredictRequest):
    session = _Session()
    try:
        # Resolve fighter names (aliases + fuzzy match)
        f1_name = FIGHTER_ALIASES.get(req.fighter1, req.fighter1)
        f2_name = FIGHTER_ALIASES.get(req.fighter2, req.fighter2)

        f1: Optional[Fighter] = _resolve_fighter(session, f1_name)
        f2: Optional[Fighter] = _resolve_fighter(session, f2_name)

        missing = [n for n, f in [(req.fighter1, f1), (req.fighter2, f2)] if not f]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Fighter(s) not found: {', '.join(missing)}"
            )

        # Market probability from odds (or default 50/50)
        if req.fighter1_odds is not None and req.fighter2_odds is not None:
            raw1 = _american_to_prob(req.fighter1_odds)
            raw2 = _american_to_prob(req.fighter2_odds)
            vig  = raw1 + raw2
            mkt_prob_f1 = raw1 / vig          # vig-normalised
        elif req.fighter1_odds is not None:
            mkt_prob_f1 = _american_to_prob(req.fighter1_odds)
        elif req.fighter2_odds is not None:
            mkt_prob_f1 = 1 - _american_to_prob(req.fighter2_odds)
        else:
            mkt_prob_f1 = 0.5

        # Feature extraction — honour as_of_date to avoid look-ahead
        as_of = datetime.combine(req.fight_date, datetime.min.time()) \
                if req.fight_date else None

        extractor = MatchupFeatureExtractor(session)
        pred = _score_row(
            session,
            extractor,
            f1.id,
            f2.id,
            mkt_prob_f1,
            as_of_date=as_of,
        )

        model_prob  = pred["model_prob_f1"]          # 0–1
        model_prob_pct = round(model_prob * 100, 1)

        # Pick is always the side the model thinks will win. We never bet
        # against our own model. Edge is signed — negative means the market
        # is more confident in our pick than the model is, in which case the
        # bet evaluator will skip with F3 (favorite low edge) or U2
        # (underdog low edge).
        if model_prob >= 0.5:
            model_pick      = req.fighter1
            pick_model_prob = model_prob
            pick_mkt_prob   = mkt_prob_f1
            pick_odds_int   = req.fighter1_odds
            pick_slot       = "fighter1"
        else:
            model_pick      = req.fighter2
            pick_model_prob = 1 - model_prob
            pick_mkt_prob   = 1 - mkt_prob_f1
            pick_odds_int   = req.fighter2_odds
            pick_slot       = "fighter2"

        edge = round((pick_model_prob - pick_mkt_prob) * 100, 1)  # signed

        # Fighter metadata — date-filtered to match feature extraction boundary
        f1_count = _fight_count_as_of(session, f1.id, as_of)
        f2_count = _fight_count_as_of(session, f2.id, as_of)

        # WMMA detection (None means unknown — treated as not-WMMA for bet rules)
        is_wmma = _matchup_wmma_flag(session, f1.id, f2.id)

        # Bet decision against the current betting_config rules
        is_favorite = pick_odds_int is not None and pick_odds_int < 0
        bet_eval = _evaluate_bet(
            fighter1_name=f1.name,
            fighter2_name=f2.name,
            pick_slot=pick_slot,
            pick_model_prob=pick_model_prob,
            pick_mkt_prob=pick_mkt_prob,
            pick_odds=pick_odds_int,
            is_favorite=is_favorite,
            is_wmma=is_wmma is True,
            f1_count=f1_count,
            f2_count=f2_count,
            as_of_date=as_of,
        )
        historical_context = describe_historical_context(
            pick_model_prob=pick_model_prob,
            pick_market_prob=pick_mkt_prob,
            pick_odds=pick_odds_int,
            is_wmma=is_wmma,
        )
        review_fields = _predict_review_fields(bet_eval)
        decision = _predict_decision_label(bet_eval)
        explanation = _predict_explanation(bet_eval)

        return {
            "fighter1":           req.fighter1,
            "fighter2":           req.fighter2,
            "model_prob_f1":      model_prob_pct,
            "model_prob_f2":      round(100 - model_prob_pct, 1),
            "model_pick":         model_pick,
            "market_prob_f1":     round(mkt_prob_f1 * 100, 1),
            "market_prob_f2":     round((1 - mkt_prob_f1) * 100, 1),
            "edge":               edge,
            "f1_odds":            req.fighter1_odds,
            "f2_odds":            req.fighter2_odds,
            "thin_data_warning":  f1_count < 3 or f2_count < 3,
            "is_wmma":            is_wmma,
            "fight_date":         req.fight_date.isoformat() if req.fight_date else None,
            "bet":                bet_eval["bet"],
            "skip_reason":        bet_eval.get("skip_reason"),
            "decision":           decision,
            "explanation":        explanation,
            "decision_source":    bet_eval.get("decision_source"),
            "review_bucket":      review_fields["review_bucket"],
            "review_tier":        review_fields["review_tier"],
            "review_label":       review_fields["review_label"],
            "pick_elo_diff":      bet_eval.get("pick_elo_diff"),
            "historical_context": historical_context,
        }

    finally:
        session.close()
