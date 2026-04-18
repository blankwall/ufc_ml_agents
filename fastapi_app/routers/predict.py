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

from database.schema import Event, Fight, Fighter
from services.predict_service import (
    FIGHTER_ALIASES,
    MatchupFeatureExtractor,
    _is_wmma,
    _load_general_model,
    _resolve_fighter,
)

router = APIRouter()

_DB_PATH     = ROOT_DIR / "data" / "ufc_database.db"
_CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"
_engine  = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)


# Skip codes (mirror events / bucket_analysis vocabulary).
SKIP_REASONS = {
    "F1":   "Favorite low confidence",
    "F2":   "Favorite odds cap exceeded",
    "U1":   "Underdog low confidence",
    "U2":   "Underdog low edge",
    "U3":   "Underdog odds cap exceeded",
    "W1":   "WMMA edge below WMMA minimum",
    "D1":   "Insufficient fight data",
    "ERR":  "Prediction failed",
}


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


def _fight_count(session, fighter_id: int, as_of=None) -> int:
    """
    Count fights for a fighter, optionally filtered to those strictly before as_of.
    Event.date is stored as "Month DD, YYYY" (not ISO), so filtering is done in
    Python after parsing — same strict-< boundary as the feature builder.
    """
    rows = (
        session.query(Event.date)
        .join(Fight, Fight.event_id == Event.id)
        .filter((Fight.fighter_1_id == fighter_id) | (Fight.fighter_2_id == fighter_id))
        .all()
    )
    if as_of is None:
        return len(rows)
    count = 0
    for (date_str,) in rows:
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str or "", fmt)
                if dt < as_of:
                    count += 1
                break
            except (ValueError, TypeError):
                continue
    return count


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
    pick_model_prob: float,   # 0–1 — model conviction on the picked side
    pick_mkt_prob:   float,   # 0–1 — implied market prob on the picked side
    pick_odds:       Optional[int],
    is_favorite:     bool,
    is_wmma:         Optional[bool],
    f1_count:        int,
    f2_count:        int,
) -> dict:
    """Apply current betting_config rules and return (bet, skip_code, reason)."""
    cfg          = _load_betting_filters()
    filters      = cfg.get("filters", {})
    wmma_rules   = cfg.get("wmma", {})

    min_fights   = filters.get("min_fights", 2)
    fav_conf     = filters.get("favorite_confidence_min", 0.65)
    ud_conf      = filters.get("underdog_confidence_min", 0.53)
    fav_cap      = filters.get("favorite_odds_cap", -300)
    ud_cap       = filters.get("underdog_odds_cap", 300)
    edge_min     = filters.get("edge_min", 0.04)
    ud_edge_min  = filters.get("underdog_edge_min", edge_min)
    wmma_min_edge = wmma_rules.get("min_edge", 0.10)

    edge_pct = pick_model_prob - pick_mkt_prob   # 0–1 scale

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
            return {"bet": False, "skip_code": "U2", "skip_reason": "Favorite low edge"}
    else:
        if pick_model_prob < ud_conf:
            return {"bet": False, "skip_code": "U1", "skip_reason": SKIP_REASONS["U1"]}
        if pick_odds is not None and pick_odds > ud_cap:
            return {"bet": False, "skip_code": "U3", "skip_reason": SKIP_REASONS["U3"]}
        if edge_pct < ud_edge_min:
            return {"bet": False, "skip_code": "U2", "skip_reason": SKIP_REASONS["U2"]}

    return {"bet": True, "skip_code": None, "skip_reason": None}


# ── scoring (as_of_date-aware version of predict_service._score_row) ─────────

def _score_fight(extractor: MatchupFeatureExtractor,
                 f1_id: int, f2_id: int,
                 market_prob_f1: float,
                 as_of_date=None) -> dict:
    import pandas as pd

    gen_model, gen_scaler, gen_features = _load_general_model()

    def _predict(fid_a: int, fid_b: int) -> float:
        feats = extractor.extract_matchup_features(fid_a, fid_b, as_of_date=as_of_date)
        feats["is_title_fight"] = 0
        X  = pd.DataFrame([feats]).reindex(columns=gen_features, fill_value=0).fillna(0)
        Xs = pd.DataFrame(gen_scaler.transform(X), columns=gen_features)
        return float(gen_model.predict_proba(Xs)[0, 1])

    gen_prob = 0.5 * (_predict(f1_id, f2_id) + (1.0 - _predict(f2_id, f1_id)))

    return {"model_prob_f1": round(gen_prob, 4), "model_source": "general"}


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
        pred = _score_fight(extractor, f1.id, f2.id, mkt_prob_f1, as_of)

        model_prob  = pred["model_prob_f1"]          # 0–1
        model_prob_pct = round(model_prob * 100, 1)
        model_pick  = req.fighter1 if model_prob >= 0.5 else req.fighter2

        # Edge: model confidence for the pick minus market probability for same fighter
        pick_model_prob = model_prob if model_prob >= 0.5 else 1 - model_prob
        pick_mkt_prob   = mkt_prob_f1 if model_prob >= 0.5 else 1 - mkt_prob_f1
        edge = round((pick_model_prob - pick_mkt_prob) * 100, 1)

        # Fighter metadata — date-filtered to match feature extraction boundary
        f1_count = _fight_count(session, f1.id, as_of)
        f2_count = _fight_count(session, f2.id, as_of)

        # WMMA detection (None means unknown — treated as not-WMMA for bet rules)
        w1 = _is_wmma(session, f1.id)
        w2 = _is_wmma(session, f2.id)
        if w1 is True or w2 is True:
            is_wmma = True
        elif w1 is None and w2 is None:
            is_wmma = None
        else:
            is_wmma = False

        # Bet decision against the current betting_config rules
        pick_odds_int = (req.fighter1_odds if model_prob >= 0.5 else req.fighter2_odds)
        is_favorite   = pick_odds_int is not None and pick_odds_int < 0
        bet_eval = _evaluate_bet(
            pick_model_prob=pick_model_prob,
            pick_mkt_prob=pick_mkt_prob,
            pick_odds=pick_odds_int,
            is_favorite=is_favorite,
            is_wmma=bool(is_wmma),
            f1_count=f1_count,
            f2_count=f2_count,
        )

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
            "fight_date":         req.fight_date.isoformat() if req.fight_date else None,
            "bet":                bet_eval["bet"],
            "skip_code":          bet_eval["skip_code"],
            "skip_reason":        bet_eval["skip_reason"],
        }

    finally:
        session.close()
