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
from database.schema import Event as _Event, Fight as _Fight
from models.utils import resolve_model_dir
from fastapi_app.services.bet_evaluator import evaluate_bet_decision
from fastapi_app.services.the_odds_api_service import get_bet_placed_map

ODDS_DIR         = ROOT_DIR / "data" / "future_fight_odds"
USER_EVENTS_DIR  = ROOT_DIR / "data" / "user_events"
OUTCOMES_CSV     = ODDS_DIR / "outcomes.csv"
CACHE_FILE       = ODDS_DIR / "predictions_cache.json"
MODEL_DIR        = ROOT_DIR / "models" / "saved"
CACHE_VERSION    = "v2"
_CONFIG_PATH     = ROOT_DIR / "config" / "betting_config.json"

UD_THRESHOLD     = 0.40   # market_prob below this → underdog blend
BLEND_WEIGHT     = 0.65   # ud_v1 weight
UNDERDOG_BLEND   = False  # set True to re-enable the underdog_v1 blend


# ── model loaders (reuse backtest_engine pattern) ────────────────────────────
_gen_model = _gen_scaler = _gen_features = None
_ud_model  = _ud_scaler  = _ud_features  = None


def _now() -> datetime:
    return datetime.now()


def _load_general_model():
    global _gen_model, _gen_scaler, _gen_features
    if _gen_model is None:
        _run_dir = resolve_model_dir(MODEL_DIR, "mar_4_v2")
        _gen_model = xgb.XGBClassifier()
        _gen_model.load_model(_run_dir / "mar_4_v2.json")
        _gen_scaler   = joblib.load(_run_dir / "mar_4_v2_feature_scaler.pkl")
        _gen_features = joblib.load(_run_dir / "mar_4_v2_feature_names.pkl")
    return _gen_model, _gen_scaler, _gen_features


def _load_underdog_model():
    global _ud_model, _ud_scaler, _ud_features
    if _ud_model is None:
        _run_dir = resolve_model_dir(MODEL_DIR, "underdog_v1")
        _ud_model = xgb.XGBClassifier()
        _ud_model.load_model(_run_dir / "underdog_v1.json")
        _ud_scaler   = joblib.load(_run_dir / "underdog_v1_feature_scaler.pkl")
        _ud_features = joblib.load(_run_dir / "underdog_v1_feature_names.pkl")
    return _ud_model, _ud_scaler, _ud_features


# ── fighter aliases (real-name → DB name or nickname stored in DB) ───────────
# Add entries here when a CSV uses a name variant the DB doesn't recognise.
FIGHTER_ALIASES: dict[str, str] = {
    "Bobby Green":            "King Green",
    "Sean Omalley":           "Sean O'Malley",
    "Charles Johson":         "Charles Johnson",
    "Michal Oleksiejczluk":   "Michal Oleksiejczuk",
    "Waldo Cortes-Acosta":    "Waldo Cortes Acosta",
    "Loneer Kavanagh":        "Lone'er Kavanagh",
    "Carlos Leal Miranda":    "Carlos Leal",
    "Long Xiao":              "Xiao Long",
    "Lupita Godinez":         "Loopy Godinez",
    "Benoit St. Denis":       "Benoit Saint Denis",
    "Benoit St Denis":        "Benoit Saint Denis",
    "Kim Sang Wook":          "Sangwook Kim",
    "Jose Medina":            "Jose Daniel Medina",
    "Montserrat Rendon":      "Montse Rendon",
    "Azamt Bekoev":           "Azamat Bekoev",
    "Casey Oneill":           "Casey O'Neill",
    "Soo Young Yoo":          "SuYoung You",
    # Outcome name mismatches (odds source vs UFC stats canonical)
    "Michael Aswell":         "Michael Aswell Jr.",
    "Cameron Rowston":        "Cam Rowston",
    "Don Mar Fan":            "Dom Mar Fan",
    "Juan Martinetti":        "Adrian Luna Martinetti",
    # Sergey sidecar / alternate-source name for the same fighter.
    "Konklak Suphisara":      "Loma Lookboonmee",
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


# ── date parsing helpers ──────────────────────────────────────────────────────

def _parse_event_date(s: str) -> Optional[datetime]:
    """
    Parse an event date string into a datetime for use as as_of_date.
    Handles full dates ("April 4th, 2026", "2026-04-04") and year-less dates
    ("April 4th", "March 21st") — for the latter the year is inferred as the
    most recent year that would place the date in the past (or present).
    Returns None for future fights (no as_of restriction needed) or unparseable strings.
    """
    import re as _re
    dt = _parse_event_date_any(s)
    return dt if dt is not None and dt <= _now() else None


def _parse_event_date_any(s: str) -> Optional[datetime]:
    """Parse an event date string without filtering out future dates."""
    import re as _re

    if not s or s.lower() in ("", "nan", "none"):
        return None
    cleaned = _re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s.strip())

    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%d %B %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    for fmt in ("%B %d", "%b %d"):
        try:
            partial = datetime.strptime(cleaned, fmt)
            return partial.replace(year=_now().year)
        except ValueError:
            continue

    try:
        return pd.to_datetime(cleaned, errors="raise").to_pydatetime()
    except Exception:
        return None


def _prediction_cache_namespace() -> str:
    """Namespace prediction cache entries by app/cache semantics version."""
    return CACHE_VERSION


def _cache_key_for_prediction(fight_key: str, *, as_of: Optional[datetime], event_date: str) -> str:
    """Use date-anchored keys for history and roll future-fight cache daily."""
    namespace = _prediction_cache_namespace()
    event_dt = _parse_event_date_any(event_date)
    if event_dt is not None and event_dt.date() > _now().date():
        return f"{namespace}|{fight_key}|future|{event_dt.strftime('%Y-%m-%d')}|{_now().strftime('%Y-%m-%d')}"

    if as_of is not None:
        return f"{namespace}|{fight_key}|{as_of.strftime('%Y-%m-%d')}"

    return f"{namespace}|{fight_key}"


def _prune_stale_future_cache(cache: dict) -> tuple[dict, bool]:
    """Drop old-namespace and prior-day future cache entries."""
    namespace_prefix = f"{_prediction_cache_namespace()}|"
    today_key = f"|{_now().strftime('%Y-%m-%d')}"
    pruned = {
        key: value
        for key, value in cache.items()
        if key.startswith(namespace_prefix) and ("|future|" not in key or key.endswith(today_key))
    }
    return pruned, len(pruned) != len(cache)


def _fight_count_as_of(session, fighter_id: int, as_of: Optional[datetime]) -> int:
    """
    Count a fighter's DB fights strictly before as_of.
    Event.date is stored as "Month DD, YYYY" (not ISO), so we filter in Python
    after parsing each date — same strict-< boundary as the feature builder.
    """
    rows = (
        session.query(_Event.date)
        .join(_Fight, _Fight.event_id == _Event.id)
        .filter((_Fight.fighter_1_id == fighter_id) | (_Fight.fighter_2_id == fighter_id))
        .all()
    )
    if as_of is None:
        return len(rows)
    count = 0
    for (date_str,) in rows:
        dt = _parse_event_date(date_str or "")
        if dt and dt < as_of:
            count += 1
    return count


def _is_wmma(session, fighter_id: int) -> Optional[bool]:
    """Check if a fighter competes in a Women's weight class.
    Returns True/False if we have data, None if unknown."""
    fight = (
        session.query(_Fight.weight_class)
        .filter((_Fight.fighter_1_id == fighter_id) | (_Fight.fighter_2_id == fighter_id))
        .filter(_Fight.weight_class.isnot(None))
        .order_by(_Fight.id.desc())
        .first()
    )
    if fight is None or fight[0] is None:
        return None
    return fight[0].startswith("Women's")


# ── single-fight prediction ───────────────────────────────────────────────────

def _score_row(session, extractor: MatchupFeatureExtractor,
               f1_id: int, f2_id: int,
               market_prob_f1: float,
               as_of_date: Optional[datetime] = None) -> dict:
    """
    Run symmetric mar_4_v2 prediction + optional underdog blend.
    as_of_date is passed to extract_matchup_features so only fight history
    strictly before that date is used — preventing look-ahead leakage.
    Returns dict with model_prob_f1, model_source.
    """
    gen_model, gen_scaler, gen_features = _load_general_model()

    def _predict_direction(fid_a: int, fid_b: int) -> float:
        feats = extractor.extract_matchup_features(fid_a, fid_b, as_of_date=as_of_date)
        feats["is_title_fight"] = 0
        X = pd.DataFrame([feats]).reindex(columns=gen_features, fill_value=0).fillna(0)
        Xs = pd.DataFrame(gen_scaler.transform(X), columns=gen_features)
        return float(gen_model.predict_proba(Xs)[0, 1])

    p_raw    = _predict_direction(f1_id, f2_id)
    p_raw_r  = _predict_direction(f2_id, f1_id)
    gen_prob = 0.5 * (p_raw + (1.0 - p_raw_r))

    if UNDERDOG_BLEND:
        # Underdog blend — applied whenever either fighter is a market underdog.
        # The underdog model expects the underdog as fighter-1, so we always call it
        # with the underdog first, then re-orient the result back to the f1/f2 frame.
        if market_prob_f1 < UD_THRESHOLD:
            # f1 is the market underdog
            try:
                ud_model, ud_scaler, ud_features = _load_underdog_model()
                feats_ud = extractor.extract_matchup_features(f1_id, f2_id, as_of_date=as_of_date)
                feats_ud["is_title_fight"] = 0
                X_ud = pd.DataFrame([feats_ud]).reindex(columns=ud_features, fill_value=0).fillna(0)
                Xs_ud = pd.DataFrame(ud_scaler.transform(X_ud), columns=ud_features)
                p_ud = float(ud_model.predict_proba(Xs_ud)[0, 1])
                model_prob = BLEND_WEIGHT * p_ud + (1 - BLEND_WEIGHT) * gen_prob
                return {"model_prob_f1": round(model_prob, 4), "model_source": "blended"}
            except Exception:
                pass
        elif (1.0 - market_prob_f1) < UD_THRESHOLD:
            # f2 is the market underdog — call the underdog model with f2 first
            try:
                ud_model, ud_scaler, ud_features = _load_underdog_model()
                feats_ud = extractor.extract_matchup_features(f2_id, f1_id, as_of_date=as_of_date)
                feats_ud["is_title_fight"] = 0
                X_ud = pd.DataFrame([feats_ud]).reindex(columns=ud_features, fill_value=0).fillna(0)
                Xs_ud = pd.DataFrame(ud_scaler.transform(X_ud), columns=ud_features)
                p_ud_f2 = float(ud_model.predict_proba(Xs_ud)[0, 1])
                gen_prob_f2 = BLEND_WEIGHT * p_ud_f2 + (1 - BLEND_WEIGHT) * (1.0 - gen_prob)
                return {"model_prob_f1": round(1.0 - gen_prob_f2, 4), "model_source": "blended"}
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

    # ── Generated The Odds API CSVs (supplemental, lower priority than manual/BFO) ──
    for csv in sorted(ODDS_DIR.glob("the_odds_api*.csv")):
        try:
            df = pd.read_csv(csv)
            df["source_file"] = csv.name
            df["source_type"] = "the_odds_api"
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
    odds = _sanitize_fighter_columns(odds)
    for col in ("event_url", "event_name", "event_date", "source_file", "source_type", "last_update"):
        if col in odds.columns:
            odds[col] = odds[col].fillna("")

    # Build fight pair key using alias-resolved names so e.g. "Bobby Green" and
    # "King Green" in the same event collapse to one canonical row.
    def _alias_key(name: str) -> str:
        resolved = FIGHTER_ALIASES.get(name.strip(), name.strip())
        return resolved.lower().strip()

    odds["_key"] = odds.apply(
        lambda r: "_vs_".join(sorted([
            _alias_key(str(r.get("fighter1", ""))),
            _alias_key(str(r.get("fighter2", ""))),
        ])), axis=1
    )

    # Pass 1: deduplicate same fight within the same event URL
    odds["_event_key"] = odds.get("event_url", odds.get("source_file", "")).astype(str) + "|" + odds["_key"]
    odds = odds.drop_duplicates("_event_key").drop(columns=["_event_key"])

    # Pass 2: deduplicate the same matchup across different event sources
    # (e.g. the same 5 fights appearing in both ufc-seattle-4018 and ufc-seattle-4095).
    # Sort by source event ID descending so the higher-numbered (newer) scrape wins.
    import re as _re
    def _src_id(src: str) -> int:
        m = _re.search(r'_(\d+)(?:\.json|\.csv)?$', str(src))
        return int(m.group(1)) if m else 0
    odds["_src_id"] = odds["source_file"].apply(_src_id)
    odds["_priority"] = odds["source_type"].map({"user_added": 3, "csv": 2, "the_odds_api": 1}).fillna(0)
    odds["_last_update"] = pd.to_datetime(odds.get("last_update"), errors="coerce", utc=True)
    odds = odds.sort_values(["_priority", "_src_id", "_last_update"], ascending=[False, False, False])
    odds = odds.drop_duplicates("_key", keep="first").drop(columns=["_key", "_src_id", "_priority", "_last_update"])

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


def _clean_fighter_name(name: str) -> str:
    """
    Strip BFO admin matchup IDs accidentally prefixed to cached fighter names.

    Example: "41970Carlos Prates" -> "Carlos Prates"
    """
    import re

    cleaned = str(name).strip()
    return re.sub(r"^\d{4,}\s*(?=[A-Z])", "", cleaned)


def _sanitize_fighter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("fighter1", "fighter2"):
        if col in df.columns:
            df[col] = df[col].map(_clean_fighter_name)
    return df


def _normalize_name(name: str) -> str:
    """Lowercase, normalise punctuation that varies across sources.
    Hyphens become spaces; apostrophes and dots are dropped."""
    import unicodedata, re
    n = _clean_fighter_name(name).lower().strip()
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


def _load_betting_filters() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        return {
            "filters": cfg.get("filters", {}),
            "wmma": cfg.get("wmma_rules", {}),
        }
    except (OSError, json.JSONDecodeError):
        return {}


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
    tracked_bets = get_bet_placed_map()
    cfg = _load_betting_filters()
    filters = cfg.get("filters", {})
    wmma_rules = cfg.get("wmma", {})

    # norm_key → real event name (outcomes have the authoritative UFC names)
    fightkey_to_ev_name: dict[str, str] = {}
    if "norm_key" in outcomes.columns and "event_name" in outcomes.columns:
        for _, orow in outcomes.iterrows():
            nk   = str(orow.get("norm_key", "")).strip()
            name = str(orow.get("event_name", "")).strip()
            if nk and name:
                fightkey_to_ev_name[nk] = name

    for _, row in odds_df.iterrows():
        def _clean_str(value) -> str:
            if pd.isna(value):
                return ""
            text = str(value).strip()
            return "" if text.lower() == "nan" else text

        f1_name     = _clean_fighter_name(str(row.get("fighter1", "")))
        f2_name     = _clean_fighter_name(str(row.get("fighter2", "")))
        ev_date     = _clean_str(row.get("event_date", ""))
        ev_url      = _clean_str(row.get("event_url", ""))
        f1_odds     = row.get("fighter1_odds")
        f2_odds     = row.get("fighter2_odds")
        # Derive raw implied probabilities from odds when available so that
        # manually-edited odds don't drift from the stored prob columns.
        # Fall back to stored probs only if odds are missing.
        def _to_raw(o):
            try:
                o = float(o)
            except (TypeError, ValueError):
                return None
            return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)
        raw1_from_odds = _to_raw(f1_odds)
        raw2_from_odds = _to_raw(f2_odds)
        if raw1_from_odds is not None and raw2_from_odds is not None:
            raw1, raw2 = raw1_from_odds, raw2_from_odds
        else:
            raw1 = float(row.get("fighter1_prob", 0.5))
            raw2 = float(row.get("fighter2_prob", 1 - raw1))
        vig      = raw1 + raw2
        mkt_prob = raw1 / vig if vig > 0 else 0.5



        source_type = _clean_str(row.get("source_type", "csv")) or "csv"
        ev_name     = _clean_str(row.get("event_name", ""))

        f1_canonical = FIGHTER_ALIASES.get(f1_name, f1_name)
        f2_canonical = FIGHTER_ALIASES.get(f2_name, f2_name)
        fkey = _fight_key(f1_canonical, f2_canonical)

        out_row = outcomes[outcomes["norm_key"] == fkey] if "norm_key" in outcomes.columns else pd.DataFrame()
        if out_row.empty and "norm_key" in outcomes.columns:
            alt_key = _fight_key(f1_name, f2_name)
            out_row = outcomes[outcomes["norm_key"] == alt_key]
        winner = str(out_row["winner"].iloc[0]).strip() if len(out_row) else None
        method = str(out_row["method"].iloc[0]).strip() if len(out_row) else None
        rd     = str(out_row["round"].iloc[0]).strip()  if len(out_row) else None

        # Use the scheduled event date for point-in-time feature extraction so
        # /events stays anchored to the card date and remains parity-safe with
        # /api/predict for the same matchup/odds/date input.
        as_of = _parse_event_date_any(ev_date)

        # Historical fights stay anchored to the event date; future fights roll
        # daily so newly ingested DB history doesn't leave /events stale.
        cache_key = _cache_key_for_prediction(fkey, as_of=as_of, event_date=ev_date)

        api_event_dt = pd.to_datetime(ev_date, errors="coerce") if ev_date else pd.NaT
        if source_type == "the_odds_api" and not pd.isna(api_event_dt):
            date_key = api_event_dt.strftime("%Y-%m-%d")
            ev_name = f"MMA Card · {date_key}"
            ev_date = date_key
            ev_url = ""
        bet_placed = tracked_bets.get((ev_date, fkey)) if source_type == "the_odds_api" and ev_date else None

        # ── model prediction (cached) ─────────────────────────────────────────
        pred = cache.get(cache_key)
        if pred is None:
            try:
                f1 = _resolve_fighter(session, f1_name)
                f2 = _resolve_fighter(session, f2_name)
                if f1 and f2:
                    pred = _score_row(session, extractor, f1.id, f2.id, mkt_prob,
                                      as_of_date=as_of)
                    pred["f1_db_name"]     = f1.name
                    pred["f2_db_name"]     = f2.name
                    pred["f1_fight_count"] = _fight_count_as_of(session, f1.id, as_of)
                    pred["f2_fight_count"] = _fight_count_as_of(session, f2.id, as_of)
                    # WMMA: True if either fighter is in a Women's division
                    w1 = _is_wmma(session, f1.id)
                    w2 = _is_wmma(session, f2.id)
                    if w1 is True or w2 is True:
                        pred["is_wmma"] = True
                    elif w1 is None and w2 is None:
                        pred["is_wmma"] = None
                    else:
                        pred["is_wmma"] = False
                else:
                    missing = [n for n, f in [(f1_name, f1), (f2_name, f2)] if not f]
                    pred = {"model_prob_f1": None, "model_source": "not_found",
                            "error": f"Fighter not found: {', '.join(missing)}"}
            except Exception as e:
                pred = {"model_prob_f1": None, "model_source": "error", "error": str(e)}
            cache[cache_key] = pred
            cache_dirty = True

        # ── compute P&L / correctness ─────────────────────────────────────────
        model_prob = pred.get("model_prob_f1")
        model_pick = correct = pnl = edge = None
        bet_eval = {
            "bet": None,
            "skip_code": None,
            "skip_reason": None,
            "decision_source": None,
            "review_bucket": None,
            "review_tier": None,
            "review_label": None,
            "pick_elo_diff": None,
        }

        if model_prob is not None:
            model_pick = f1_name if model_prob >= 0.5 else f2_name
            # Edge from the *picked* fighter's perspective (matches /api/predict)
            pick_model_prob = model_prob if model_prob >= 0.5 else 1 - model_prob
            pick_mkt_prob   = mkt_prob   if model_prob >= 0.5 else 1 - mkt_prob
            edge = round((pick_model_prob - pick_mkt_prob) * 100, 1)
            pick_odds_int = int(f1_odds) if model_prob >= 0.5 and pd.notna(f1_odds) else (
                int(f2_odds) if model_prob < 0.5 and pd.notna(f2_odds) else None
            )
            pick_slot = "fighter1" if model_prob >= 0.5 else "fighter2"
            if pred.get("f1_db_name") and pred.get("f2_db_name"):
                bet_eval = evaluate_bet_decision(
                    fighter1_name=pred["f1_db_name"],
                    fighter2_name=pred["f2_db_name"],
                    pick_slot=pick_slot,
                    pick_model_prob=pick_model_prob,
                    pick_mkt_prob=pick_mkt_prob,
                    pick_odds=pick_odds_int,
                    is_favorite=pick_odds_int is not None and pick_odds_int < 0,
                    is_wmma=pred.get("is_wmma") is True,
                    f1_count=pred.get("f1_fight_count", 0),
                    f2_count=pred.get("f2_fight_count", 0),
                    filters=filters,
                    wmma_rules=wmma_rules,
                    as_of_date=as_of,
                )

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
            "bet_placed":     bet_placed,
            "f1_fight_count": pred.get("f1_fight_count"),
            "f2_fight_count": pred.get("f2_fight_count"),
            "is_wmma":        pred.get("is_wmma"),
            "bet":            bet_eval.get("bet"),
            "skip_code":      bet_eval.get("skip_code"),
            "skip_reason":    bet_eval.get("skip_reason"),
            "decision_source": bet_eval.get("decision_source"),
            "review_bucket":  bet_eval.get("review_bucket"),
            "review_tier":    bet_eval.get("review_tier"),
            "review_label":   bet_eval.get("review_label"),
            "pick_elo_diff":  bet_eval.get("pick_elo_diff"),
        }

        ev_name_real = fightkey_to_ev_name.get(fkey, ev_name)
        ev_key = f"the_odds_api|{ev_date}" if source_type == "the_odds_api" and ev_date else (ev_url or ev_name)
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
        if not d:
            return datetime.min
        cleaned = d.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        for fmt in ("%B %d", "%B %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(cleaned, fmt)
            except Exception:
                continue
        return datetime.min

    events.sort(key=lambda e: _parse_date(e["event_date"]))
    return events


# ── public entry points ───────────────────────────────────────────────────────

def get_events_data() -> list[dict]:
    """All events (CSVs + user-added) with predictions. Used by /api/events."""
    odds_df  = _load_all_odds()
    outcomes = _load_outcomes()
    cache, cache_dirty = _prune_stale_future_cache(_load_cache())
    if odds_df.empty:
        return []

    session   = _open_session()
    extractor = MatchupFeatureExtractor(session)
    try:
        events_map, prediction_cache_dirty = _run_prediction_loop(odds_df, outcomes, cache, session, extractor)
    finally:
        session.close()

    if cache_dirty or prediction_cache_dirty:
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
    odds_df = _sanitize_fighter_columns(odds_df)

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

    cache, cache_dirty = _prune_stale_future_cache(_load_cache())
    session   = _open_session()
    extractor = MatchupFeatureExtractor(session)
    try:
        events_map, prediction_cache_dirty = _run_prediction_loop(odds_df, outcomes_df, cache, session, extractor)
    finally:
        session.close()

    if cache_dirty or prediction_cache_dirty:
        _save_cache(cache)

    events = _attach_summaries(events_map)
    # Return the single event dict (there should be exactly one)
    return events[0] if events else {"error": "No fights predicted"}
