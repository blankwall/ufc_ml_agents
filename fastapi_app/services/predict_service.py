"""
Predict Service
===============
Runs mar_4_v2 + underdog blend predictions for fights defined in
data/future_fight_odds/*.csv, joins with outcome data from outcomes.csv,
and caches results to avoid re-running the slow feature extraction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# ── path setup so we can import from the repo root ───────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from features.matchup_features import MatchupFeatureExtractor

ODDS_DIR         = ROOT_DIR / "data" / "future_fight_odds"
USER_EVENTS_DIR  = ROOT_DIR / "data" / "user_events"
OUTCOMES_CSV     = ODDS_DIR / "outcomes.csv"
CACHE_FILE       = ODDS_DIR / "predictions_cache.json"
MODEL_DIR        = ROOT_DIR / "models" / "saved"

UD_THRESHOLD  = 0.40   # market_prob below this → underdog blend
BLEND_WEIGHT  = 0.65   # ud_v1 weight


# ── model loaders (reuse backtest_engine pattern) ────────────────────────────
_gen_model = _gen_scaler = _gen_features = None
_ud_model  = _ud_scaler  = _ud_features  = None


def _load_general_model():
    global _gen_model, _gen_scaler, _gen_features
    if _gen_model is None:
        _gen_model = xgb.XGBClassifier()
        _gen_model.load_model(MODEL_DIR / "mar_4_v2.json")
        _gen_scaler   = joblib.load(MODEL_DIR / "mar_4_v2_feature_scaler.pkl")
        _gen_features = joblib.load(MODEL_DIR / "mar_4_v2_feature_names.pkl")
    return _gen_model, _gen_scaler, _gen_features


def _load_underdog_model():
    global _ud_model, _ud_scaler, _ud_features
    if _ud_model is None:
        _ud_model = xgb.XGBClassifier()
        _ud_model.load_model(MODEL_DIR / "underdog_v1.json")
        _ud_scaler   = joblib.load(MODEL_DIR / "underdog_v1_feature_scaler.pkl")
        _ud_features = joblib.load(MODEL_DIR / "underdog_v1_feature_names.pkl")
    return _ud_model, _ud_scaler, _ud_features


# ── fighter aliases (real-name → DB name or nickname stored in DB) ───────────
# Add entries here when a CSV uses a name variant the DB doesn't recognise.
FIGHTER_ALIASES: dict[str, str] = {
    "Bobby Green":            "King Green",
    "Sean Omalley":           "Sean O'Malley",
    "Charles Johson":         "Charles Johnson",   # CSV typo
    "Michal Oleksiejczluk":   "Michal Oleksiejczuk",  # CSV typo
    "Waldo Cortes-Acosta":    "Waldo Cortes Acosta",
    "Loneer Kavanagh":        "Lone'er Kavanagh",
    "Carlos Leal Miranda":    "Carlos Leal",
}


# ── fighter resolver ──────────────────────────────────────────────────────────

def _resolve_fighter(session, name: str):
    """
    Best-effort fighter name → DB Fighter.
    Normalises apostrophes, hyphens, and dots on both sides so e.g.
    'Loneer Kavanagh' matches "Lone'er Kavanagh" and
    'Waldo Cortes-Acosta' matches 'Waldo Cortes Acosta'.
    """
    import re as _re
    from sqlalchemy import or_, text
    from database.schema import Fighter, Fight

    def _best(rows):
        """From a list of Fighter objects pick the one with most DB fights."""
        if len(rows) == 1:
            return rows[0]
        scored = []
        for f in rows:
            cnt = session.query(Fight).filter(
                or_(Fight.fighter_1_id == f.id, Fight.fighter_2_id == f.id)
            ).count()
            scored.append((cnt, f.id, f))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0][2]

    def _strip(s: str) -> str:
        """Remove punctuation that varies between sources."""
        return _re.sub(r"['\.\-]", "", s).strip()

    # Strategy 0: known aliases
    lookup_name = FIGHTER_ALIASES.get(name, name)

    # Strategy 1: standard ilike (exact punctuation)
    rows = session.query(Fighter).filter(Fighter.name.ilike(f"%{lookup_name}%")).all()
    if rows:
        return _best(rows)

    # Strategy 2: SQL-level normalisation — strip apostrophes/dots, replace hyphens with space
    def _norm_sql(s: str) -> str:
        return _re.sub(r"['\.]", "", s.replace("-", " ")).strip().lower()

    norm = _norm_sql(lookup_name)
    if norm:
        sql = text(
            "SELECT id FROM fighters "
            "WHERE LOWER(REPLACE(REPLACE(name, '''', ''), '.', '')) "
            "LIKE :q"
        )
        ids = [r[0] for r in session.execute(sql, {"q": f"%{norm}%"})]
        if ids:
            rows = session.query(Fighter).filter(Fighter.id.in_(ids)).all()
            if rows:
                return _best(rows)

    # Strategy 3: first name + prefix of last name (handles typos / extra letters)
    parts = lookup_name.split()
    if len(parts) >= 2:
        first = _re.sub(r"['\.\-]", "", parts[0]).lower()
        # Try progressively shorter last-name prefixes (min 4 chars)
        raw_last = _re.sub(r"['\.\-]", "", " ".join(parts[1:])).lower()
        for prefix_len in range(min(len(raw_last), 8), 3, -1):
            prefix = raw_last[:prefix_len]
            sql = text(
                "SELECT id FROM fighters "
                "WHERE LOWER(REPLACE(REPLACE(name, '''', ''), '-', ' ')) LIKE :q"
            )
            ids = [r[0] for r in session.execute(sql, {"q": f"%{prefix}%"})]
            if ids:
                rows = session.query(Fighter).filter(Fighter.id.in_(ids)).all()
                rows = [f for f in rows if first[:3] in f.name.lower().replace("'","").replace("-"," ")]
                if rows:
                    return _best(rows)

    return None


# ── single-fight prediction ───────────────────────────────────────────────────

def _score_row(session, extractor: MatchupFeatureExtractor,
               f1_id: int, f2_id: int,
               market_prob_f1: float) -> dict:
    """
    Run symmetric mar_4_v2 prediction + optional underdog blend.
    Returns dict with model_prob_f1, model_source.
    """
    gen_model, gen_scaler, gen_features = _load_general_model()

    def _predict_direction(fid_a: int, fid_b: int) -> float:
        feats = extractor.extract_matchup_features(fid_a, fid_b)
        feats["is_title_fight"] = 0
        X = pd.DataFrame([feats]).reindex(columns=gen_features, fill_value=0).fillna(0)
        Xs = pd.DataFrame(gen_scaler.transform(X), columns=gen_features)
        return float(gen_model.predict_proba(Xs)[0, 1])

    p_raw    = _predict_direction(f1_id, f2_id)
    p_raw_r  = _predict_direction(f2_id, f1_id)
    gen_prob = 0.5 * (p_raw + (1.0 - p_raw_r))

    # Underdog blend when f1 is the market underdog
    if market_prob_f1 < UD_THRESHOLD:
        try:
            ud_model, ud_scaler, ud_features = _load_underdog_model()
            feats_ud = extractor.extract_matchup_features(f1_id, f2_id)
            feats_ud["is_title_fight"] = 0
            X_ud = pd.DataFrame([feats_ud]).reindex(columns=ud_features, fill_value=0).fillna(0)
            Xs_ud = pd.DataFrame(ud_scaler.transform(X_ud), columns=ud_features)
            p_ud = float(ud_model.predict_proba(Xs_ud)[0, 1])
            model_prob = BLEND_WEIGHT * p_ud + (1 - BLEND_WEIGHT) * gen_prob
            return {"model_prob_f1": round(model_prob, 4), "model_source": "blended"}
        except Exception:
            pass

    return {"model_prob_f1": round(gen_prob, 4), "model_source": "general"}


# ── CSV / outcomes loading ────────────────────────────────────────────────────

def _load_all_odds() -> pd.DataFrame:
    """Read and deduplicate all per-event odds CSVs + user-added events."""
    frames = []

    # ── Static CSVs ───────────────────────────────────────────────────────────
    for csv in sorted(ODDS_DIR.glob("ufc*.csv")):
        if csv.name in ("all_events.csv",):
            continue
        try:
            df = pd.read_csv(csv)
            df["source_file"] = csv.name
            df["source_type"] = "csv"
            frames.append(df)
        except Exception:
            pass

    # ── User-added events (data/user_events/*.json) ───────────────────────────
    if USER_EVENTS_DIR.exists():
        for jf in sorted(USER_EVENTS_DIR.glob("*.json")):
            try:
                payload = json.loads(jf.read_text())
                fight_rows = payload.get("fights", [])
                if not fight_rows:
                    continue
                df = pd.DataFrame(fight_rows)
                # BFO scraper uses fighter1_odds/fighter2_odds + fighter1_prob/fighter2_prob
                # which matches the CSV schema — just mark source_type
                df["source_file"]  = jf.name
                df["source_type"]  = "user_added"
                frames.append(df)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame()

    odds = pd.concat(frames, ignore_index=True)

    # Normalise column names
    odds.columns = [c.strip().lower() for c in odds.columns]

    # Deduplicate: same pair on same event  (handles Bobby Green / King Green dup)
    odds["_key"] = odds.apply(
        lambda r: "_vs_".join(sorted([
            str(r.get("fighter1", "")).lower().strip(),
            str(r.get("fighter2", "")).lower().strip(),
        ])), axis=1
    )
    odds["_event_key"] = odds.get("event_url", odds.get("source_file", "")).astype(str) + "|" + odds["_key"]
    odds = odds.drop_duplicates("_event_key").drop(columns=["_key", "_event_key"])

    return odds.reset_index(drop=True)


def _load_outcomes() -> pd.DataFrame:
    frames = []

    # ── Static outcomes.csv ───────────────────────────────────────────────────
    if OUTCOMES_CSV.exists():
        df = pd.read_csv(OUTCOMES_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)

    # ── Outcomes embedded in user-added event JSONs ───────────────────────────
    if USER_EVENTS_DIR.exists():
        for jf in sorted(USER_EVENTS_DIR.glob("*.json")):
            try:
                payload = json.loads(jf.read_text())
                outcome_rows = payload.get("outcomes", [])
                if outcome_rows:
                    df = pd.DataFrame(outcome_rows)
                    df.columns = [c.strip().lower() for c in df.columns]
                    frames.append(df)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame(columns=["fighter1", "fighter2", "winner", "method", "round",
                                     "fight_key", "norm_key", "event_name"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="fight_key", keep="last") if "fight_key" in out.columns else out
    out["norm_key"] = out.apply(
        lambda r: _fight_key(str(r.get("fighter1","")), str(r.get("fighter2",""))), axis=1
    )
    return out


def _normalize_name(name: str) -> str:
    """Lowercase, normalise punctuation that varies across sources.
    Hyphens become spaces; apostrophes and dots are dropped."""
    import unicodedata, re
    n = name.lower().strip()
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = n.replace("-", " ")          # hyphen → space (Cortes-Acosta → Cortes Acosta)
    n = re.sub(r"['\.`]", "", n)     # drop apostrophes and dots
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def _fight_key(f1: str, f2: str) -> str:
    return "_vs_".join(sorted([_normalize_name(f1), _normalize_name(f2)]))


# ── cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── shared DB session factory ─────────────────────────────────────────────────

def _open_session():
    DB_ABS  = ROOT_DIR / "data" / "ufc_database.db"
    engine  = create_engine(f"sqlite:///{DB_ABS}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    return Session()


# ── core prediction loop (shared by get_events_data and analyze_event) ────────

def _run_prediction_loop(
    odds_df: pd.DataFrame,
    outcomes: pd.DataFrame,
    cache: dict,
    session,
    extractor,
) -> tuple[dict, bool]:
    """
    Iterate over odds_df rows, run model predictions (cached), join outcomes,
    and group fights by event URL.

    Returns (events_map, cache_dirty).
    """
    cache_dirty = False
    events_map: dict[str, dict] = {}

    # norm_key → real event name (outcomes have the authoritative UFC names)
    fightkey_to_ev_name: dict[str, str] = {}
    if "norm_key" in outcomes.columns and "event_name" in outcomes.columns:
        for _, orow in outcomes.iterrows():
            nk   = str(orow.get("norm_key", "")).strip()
            name = str(orow.get("event_name", "")).strip()
            if nk and name:
                fightkey_to_ev_name[nk] = name

    for _, row in odds_df.iterrows():
        f1_name     = str(row.get("fighter1", "")).strip()
        f2_name     = str(row.get("fighter2", "")).strip()
        ev_date     = str(row.get("event_date", "")).strip()
        ev_url      = str(row.get("event_url", "")).strip()
        f1_odds     = row.get("fighter1_odds")
        f2_odds     = row.get("fighter2_odds")
        mkt_prob    = float(row.get("fighter1_prob", 0.5))
        source_type = str(row.get("source_type", "csv"))
        ev_name     = str(row.get("event_name", "")).strip()

        f1_canonical = FIGHTER_ALIASES.get(f1_name, f1_name)
        f2_canonical = FIGHTER_ALIASES.get(f2_name, f2_name)
        fkey = _fight_key(f1_canonical, f2_canonical)

        # ── model prediction (cached) ─────────────────────────────────────────
        pred = cache.get(fkey)
        if pred is None:
            try:
                f1 = _resolve_fighter(session, f1_name)
                f2 = _resolve_fighter(session, f2_name)
                if f1 and f2:
                    pred = _score_row(session, extractor, f1.id, f2.id, mkt_prob)
                    pred["f1_db_name"] = f1.name
                    pred["f2_db_name"] = f2.name
                else:
                    missing = [n for n, f in [(f1_name, f1), (f2_name, f2)] if not f]
                    pred = {"model_prob_f1": None, "model_source": "not_found",
                            "error": f"Fighter not found: {', '.join(missing)}"}
            except Exception as e:
                pred = {"model_prob_f1": None, "model_source": "error", "error": str(e)}
            cache[fkey] = pred
            cache_dirty = True

        # ── join outcome ──────────────────────────────────────────────────────
        out_row = outcomes[outcomes["norm_key"] == fkey] if "norm_key" in outcomes.columns else pd.DataFrame()
        if out_row.empty and "norm_key" in outcomes.columns:
            alt_key = _fight_key(f1_name, f2_name)
            out_row = outcomes[outcomes["norm_key"] == alt_key]
        winner = str(out_row["winner"].iloc[0]).strip() if len(out_row) else None
        method = str(out_row["method"].iloc[0]).strip() if len(out_row) else None
        rd     = str(out_row["round"].iloc[0]).strip()  if len(out_row) else None

        # ── compute P&L / correctness ─────────────────────────────────────────
        model_prob = pred.get("model_prob_f1")
        model_pick = correct = pnl = edge = None

        if model_prob is not None:
            model_pick = f1_name if model_prob >= 0.5 else f2_name
            edge = round((model_prob - mkt_prob) * 100, 1)

            if winner:
                w_norm  = _normalize_name(winner)
                f1_norm = _normalize_name(f1_name)
                f2_norm = _normalize_name(f2_name)
                correct = (
                    (model_prob >= 0.5 and (f1_norm in w_norm or w_norm in f1_norm)) or
                    (model_prob <  0.5 and (f2_norm in w_norm or w_norm in f2_norm))
                )
                odds_for_bet = float(f1_odds if model_prob >= 0.5 else f2_odds) \
                               if (f1_odds if model_prob >= 0.5 else f2_odds) else None
                if odds_for_bet is not None:
                    pnl = round(
                        100 * (odds_for_bet / 100 if odds_for_bet > 0 else 100 / abs(odds_for_bet))
                        if correct else -100.0,
                        1,
                    )

        fight = {
            "fighter1":       f1_name,
            "fighter2":       f2_name,
            "f1_odds":        int(f1_odds) if pd.notna(f1_odds) else None,
            "f2_odds":        int(f2_odds) if pd.notna(f2_odds) else None,
            "market_prob_f1": round(mkt_prob * 100, 1),
            "model_prob_f1":  round(model_prob * 100, 1) if model_prob is not None else None,
            "model_source":   pred.get("model_source"),
            "model_pick":     model_pick,
            "edge":           edge,
            "winner":         winner,
            "method":         method,
            "round":          rd,
            "correct":        correct,
            "pnl":            pnl,
            "error":          pred.get("error"),
            "source_type":    source_type,
        }

        ev_name_real = fightkey_to_ev_name.get(fkey, ev_name)
        ev_key = ev_url or ev_name
        if ev_key not in events_map:
            events_map[ev_key] = {
                "event_name":  ev_name_real,
                "event_date":  ev_date,
                "event_url":   ev_url,
                "source_type": source_type,
                "fights":      [],
            }
        elif ev_name_real != "UFC" and events_map[ev_key]["event_name"] == "UFC":
            events_map[ev_key]["event_name"] = ev_name_real
        events_map[ev_key]["fights"].append(fight)

    return events_map, cache_dirty


def _attach_summaries(events_map: dict) -> list[dict]:
    """Add per-event summary stats and return a sorted list."""
    events = []
    for ev in events_map.values():
        fights      = ev["fights"]
        with_result = [f for f in fights if f["correct"] is not None]
        wins        = sum(1 for f in with_result if f["correct"])
        total_pnl   = sum(f["pnl"] for f in with_result if f["pnl"] is not None)
        n           = len(with_result)
        ev["summary"] = {
            "n_fights":  len(fights),
            "n_results": n,
            "wins":      wins,
            "accuracy":  round(wins / n * 100, 1) if n else None,
            "pnl":       round(total_pnl, 1),
            "roi":       round(total_pnl / (n * 100) * 100, 1) if n else None,
        }
        events.append(ev)

    def _parse_date(d: str) -> datetime:
        try:
            return datetime.strptime(
                d.replace("st","").replace("nd","").replace("rd","").replace("th",""), "%B %d"
            )
        except Exception:
            return datetime.min

    events.sort(key=lambda e: _parse_date(e["event_date"]))
    return events


# ── public entry points ───────────────────────────────────────────────────────

def get_events_data() -> list[dict]:
    """All events (CSVs + user-added) with predictions. Used by /api/events."""
    odds_df  = _load_all_odds()
    outcomes = _load_outcomes()
    cache    = _load_cache()
    if odds_df.empty:
        return []

    session   = _open_session()
    extractor = MatchupFeatureExtractor(session)
    try:
        events_map, cache_dirty = _run_prediction_loop(odds_df, outcomes, cache, session, extractor)
    finally:
        session.close()

    if cache_dirty:
        _save_cache(cache)

    return _attach_summaries(events_map)


def analyze_event(bfo_url: str, ufc_stats_url: Optional[str] = None) -> dict:
    """
    One-shot: scrape (or load cached) a BFO event, run model predictions,
    and return a single event dict with full fight analysis.

    Used by POST /api/analyze.
    """
    import re as _re
    from services.scraper_service import scrape_and_save, USER_EVENTS_DIR, _slug

    slug     = _slug(bfo_url)
    ev_path  = USER_EVENTS_DIR / f"{slug}.json"

    # Scrape + save if not already on disk
    if not ev_path.exists():
        scrape_and_save(bfo_url, ufc_stats_url)

    payload      = json.loads(ev_path.read_text())
    fight_rows   = payload.get("fights", [])
    outcome_rows = payload.get("outcomes", [])

    if not fight_rows:
        raise ValueError(f"No fights found for {bfo_url}")

    # Build minimal DataFrames (same schema as the CSV pipeline)
    odds_df = pd.DataFrame(fight_rows)
    odds_df.columns = [c.strip().lower() for c in odds_df.columns]

    if outcome_rows:
        outcomes_df = pd.DataFrame(outcome_rows)
        outcomes_df.columns = [c.strip().lower() for c in outcomes_df.columns]
        if "fight_key" in outcomes_df.columns:
            outcomes_df = outcomes_df.drop_duplicates(subset="fight_key", keep="last")
        outcomes_df["norm_key"] = outcomes_df.apply(
            lambda r: _fight_key(str(r.get("fighter1","")), str(r.get("fighter2",""))), axis=1
        )
    else:
        outcomes_df = pd.DataFrame(columns=["fighter1","fighter2","winner","method","round",
                                            "fight_key","norm_key","event_name"])

    cache     = _load_cache()
    session   = _open_session()
    extractor = MatchupFeatureExtractor(session)
    try:
        events_map, cache_dirty = _run_prediction_loop(odds_df, outcomes_df, cache, session, extractor)
    finally:
        session.close()

    if cache_dirty:
        _save_cache(cache)

    events = _attach_summaries(events_map)
    # Return the single event dict (there should be exactly one)
    return events[0] if events else {"error": "No fights predicted"}
