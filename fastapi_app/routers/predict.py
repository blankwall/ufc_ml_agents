"""
POST /api/predict
-----------------
One-off fight prediction given two fighter names, an optional fight date,
and optional American odds.  No caching — runs fresh on every call.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtest.confidence_profile import describe_confidence
from database.schema import Fighter
from services.bet_evaluator import SKIP_REASONS, evaluate_bet_decision
from services.predict_context_service import build_predict_context
from services.predict_service import (
    FIGHTER_ALIASES,
    MatchupFeatureExtractor,
    _fight_count_as_of,
    _is_wmma,
    _prediction_cutoff_datetime,
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
    fight_date:    Optional[date] = None   # freezes prediction to the prior-day snapshot
    fighter1_odds: Optional[int]  = None   # American odds, e.g. -380 or +310
    fighter2_odds: Optional[int]  = None


class PredictContextRequest(PredictRequest):
    pass


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
    as_of=None,
    session=None,
    snapshot_cache=None,
) -> dict:
    """Apply current betting_config rules and return (bet, skip_code, reason)."""
    if session is None:
        cfg = _load_betting_filters()
        filters = cfg.get("filters", {})
        wmma_rules = cfg.get("wmma", {})
        min_fights = filters.get("min_fights", 2)
        fav_conf = filters.get("favorite_confidence_min", 0.65)
        ud_conf = filters.get("underdog_confidence_min", 0.53)
        fav_cap = filters.get("favorite_odds_cap", -300)
        ud_cap = filters.get("underdog_odds_cap", 300)
        edge_min = filters.get("edge_min", 0.04)
        ud_edge_min = filters.get("underdog_edge_min", edge_min)
        wmma_min_edge = wmma_rules.get("min_edge", 0.10)
        edge_pct = pick_model_prob - pick_mkt_prob

        if f1_count < min_fights or f2_count < min_fights:
            return {"bet": False, "skip_code": "D1", "skip_reason": SKIP_REASONS["D1"]}
        if is_wmma and edge_pct < wmma_min_edge:
            return {"bet": False, "skip_code": "W1", "skip_reason": SKIP_REASONS["W1"]}
        if is_favorite:
            if pick_model_prob < fav_conf:
                return {"bet": False, "skip_code": "F1", "skip_reason": SKIP_REASONS["F1"]}
            if pick_odds is not None and pick_odds < fav_cap:
                return {"bet": False, "skip_code": "F2", "skip_reason": SKIP_REASONS["F2"]}
            if edge_pct < edge_min:
                return {"bet": False, "skip_code": "F3", "skip_reason": SKIP_REASONS["F3"]}
        else:
            if pick_model_prob < ud_conf:
                return {"bet": False, "skip_code": "U1", "skip_reason": SKIP_REASONS["U1"]}
            if pick_odds is not None and pick_odds > ud_cap:
                return {"bet": False, "skip_code": "U3", "skip_reason": SKIP_REASONS["U3"]}
            if edge_pct < ud_edge_min:
                return {"bet": False, "skip_code": "U2", "skip_reason": SKIP_REASONS["U2"]}
        return {"bet": True, "skip_code": None, "skip_reason": None}

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
        as_of=as_of,
        session=session,
        snapshot_cache=snapshot_cache,
    )


def _matchup_wmma_flag(session, fighter1_id: int, fighter2_id: int) -> Optional[bool]:
    w1 = _is_wmma(session, fighter1_id)
    w2 = _is_wmma(session, fighter2_id)
    if w1 is True or w2 is True:
        return True
    if w1 is None and w2 is None:
        return None
    return False


@router.post("/predict/context")
async def predict_fight_context(req: PredictContextRequest):
    session = _Session()
    try:
        return build_predict_context(
            fighter1=req.fighter1,
            fighter2=req.fighter2,
            fight_date=req.fight_date,
            fighter1_odds=req.fighter1_odds,
            fighter2_odds=req.fighter2_odds,
            session=session,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


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

        # Explicit fight dates are evaluated from the previous day's snapshot.
        as_of = _prediction_cutoff_datetime(req.fight_date)

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
            as_of=as_of,
            session=session,
        )
        confidence = describe_confidence(pick_model_prob)

        return {
            "fighter1":           req.fighter1,
            "fighter2":           req.fighter2,
            "fighter1_db_name":   f1.name,
            "fighter2_db_name":   f2.name,
            "model_prob_f1":      model_prob_pct,
            "model_prob_f2":      round(100 - model_prob_pct, 1),
            "model_source":       pred["model_source"],
            "model_pick":         model_pick,
            "market_prob_f1":     round(mkt_prob_f1 * 100, 1),
            "market_prob_f2":     round((1 - mkt_prob_f1) * 100, 1),
            "edge":               edge,
            "f1_odds":            req.fighter1_odds,
            "f2_odds":            req.fighter2_odds,
            "f1_fight_count":     f1_count,
            "f2_fight_count":     f2_count,
            "f1_record":          f"{f1.wins}-{f1.losses}-{f1.draws}",
            "f2_record":          f"{f2.wins}-{f2.losses}-{f2.draws}",
            "thin_data_warning":  f1_count < 3 or f2_count < 3,
            "is_wmma":            is_wmma,
            "confidence_score":   confidence["confidence_score"],
            "confidence_historical_win_rate": confidence["confidence_historical_win_rate"],
            "fight_date":         req.fight_date.isoformat() if req.fight_date else None,
            "bet":                bet_eval["bet"],
            "skip_code":          bet_eval["skip_code"],
            "skip_reason":        bet_eval["skip_reason"],
            "decision_source":    bet_eval.get("decision_source"),
            "review_candidate":   bet_eval.get("review_candidate", False),
            "review_bucket":      bet_eval.get("review_bucket"),
            "review_label":       bet_eval.get("review_label"),
            "review_reason":      bet_eval.get("review_reason"),
            "signal_bucket":      bet_eval.get("signal_bucket"),
            "signal_label":       bet_eval.get("signal_label"),
            "signal_reason":      bet_eval.get("signal_reason"),
            "signal_tags":        bet_eval.get("signal_tags", []),
            "pick_elo_diff":      bet_eval.get("pick_elo_diff"),
            "cardio_score_diff":  bet_eval.get("cardio_score_diff"),
        }

    finally:
        session.close()
