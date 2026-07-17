#!/usr/bin/env python3
"""
Backtest UFC predictions against betting odds.
Run predictions on all fights and calculate expected value.

Odds can come from:
  - A CSV (e.g. backtest/odds/ufc_2025_odds.csv or export from DB)
  - Export from DB: python scripts/export_odds_from_db.py --year 2025 -o backtest/odds/db_2025.csv

Usage:
  python backtest/backtest_2025.py
  python backtest/backtest_2025.py --config backtest/backtest_config.json
  python backtest/backtest_2025.py --odds backtest/odds/ufc_2025_odds.csv --model mar_4_v2
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from database.db_manager import DatabaseManager

# Default paths
DEFAULT_ODDS_CSV = PROJECT_ROOT / "backtest" / "odds" / "ufc_2025_odds.csv"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "backtest_config.json"


def default_results_path_for_odds(odds_path: Path) -> Path:
    """Infer the canonical results filename from the odds input year."""
    name = odds_path.name
    if "2026" in name:
        return Path(__file__).resolve().parent / "backtest_2026_results.csv"
    if "2025" in name:
        return Path(__file__).resolve().parent / "backtest_2025_results.csv"
    return Path(__file__).resolve().parent / "backtest_results.csv"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _first_present(row: pd.Series, *columns: str, default=None):
    for column in columns:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return value
    return default


def odds_provenance(row: pd.Series, *, odds_path: Path, source_line: int) -> dict:
    """Carry optional odds source metadata through the generated results CSV."""
    optional_fields = {
        "odds_source_file": _first_present(row, "odds_source_file", "source_file", default=_relative_path(odds_path)),
        "odds_source_line": _first_present(row, "odds_source_line", "source_row", default=source_line),
        "odds_source_type": _first_present(row, "odds_source_type", "source_type"),
        "odds_source_row": _first_present(row, "odds_source_row", "source_key"),
        "source_event_id": _first_present(row, "source_event_id"),
        "source_url": _first_present(row, "source_url"),
        "scraped_at": _first_present(row, "scraped_at"),
        "bookmaker": _first_present(row, "bookmaker"),
        "odds_timestamp": _first_present(row, "odds_timestamp"),
        "odds_is_opening_line": _first_present(row, "odds_is_opening_line", "is_opening_line"),
        "odds_is_closing_line": _first_present(row, "odds_is_closing_line", "is_closing_line"),
    }
    return {key: (None if pd.isna(value) else value) for key, value in optional_fields.items()}

def load_config(config_path: Path | str) -> dict:
    """Load backtest config from JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config not found: {config_path}, using defaults")
        return {}
    with open(config_path) as f:
        return json.load(f)


# ── Underdog blend constants (must match fastapi_app/services/predict_service.py) ──
UD_THRESHOLD  = 0.40   # market implied prob below this → underdog model blended in
BLEND_WEIGHT  = 0.65   # underdog_v1 weight in the blend

MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# Module-level lazy caches so models are only loaded once per run
_model_cache: dict = {}


def _load_model(model_name: str) -> tuple:
    """Lazily load and cache an XGBoost model + scaler + feature list."""
    if model_name not in _model_cache:
        import joblib
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_DIR / f"{model_name}.json")
        scaler   = joblib.load(MODEL_DIR / f"{model_name}_feature_scaler.pkl")
        features = joblib.load(MODEL_DIR / f"{model_name}_feature_names.pkl")
        _model_cache[model_name] = (clf, scaler, features)
    return _model_cache[model_name]


def _predict_one_direction(session, f1_id: int, f2_id: int,
                           model_name: str, as_of_date: str | None) -> float:
    """Return P(f1 wins) using a single model in one fighter order."""
    from features.matchup_features import MatchupFeatureExtractor
    clf, scaler, features = _load_model(model_name)
    extractor = MatchupFeatureExtractor(session)
    feats = extractor.extract_matchup_features(f1_id, f2_id, as_of_date=as_of_date)
    feats["is_title_fight"] = 0
    X  = pd.DataFrame([feats]).reindex(columns=features, fill_value=0).fillna(0)
    Xs = pd.DataFrame(scaler.transform(X), columns=features)
    return float(clf.predict_proba(Xs)[0, 1])


# Known name mismatches between odds sources and the DB
_NAME_FIXES: dict[str, str] = {
    "sean omalley":           "Sean O'Malley",
    "waldo cortes-acosta":    "Waldo Cortes Acosta",
    "charles johson":         "Charles Johnson",
    "kim sang wook":          "Sangwook Kim",
    "michal oleksiejczluk":   "Michal Oleksiejczuk",
    "carlos leal miranda":    "Carlos Leal",
    "loneer kavanagh":        "Lone'er Kavanagh",
    "jose medina":            "Jose Daniel Medina",
    "bobby green":            "King Green",
    "long xiao":              "Xiao Long",
    "montserrat rendon":      "Montse Rendon",
    "soo young yoo":          "SuYoung You",
    "casey oneill":           "Casey O'Neill",
    "azamt bekoev":           "Azamat Bekoev",
    "lupita godinez":         "Loopy Godinez",
    "benoit st. denis":       "Benoit Saint Denis",
    "benoit st denis":        "Benoit Saint Denis",
    "michael aswell":         "Michael Aswell Jr.",
    "cameron rowston":        "Cam Rowston",
    "don mar fan":            "Dom Mar Fan",
    # Sergey sidecar / alternate-source name for the same fighter.
    "konklak suphisara":      "Loma Lookboonmee",
}


def _normalize_name(name: str) -> str:
    """Lowercase and strip punctuation that varies between data sources (apostrophes, periods)."""
    return name.lower().replace("'", "").replace(".", "").replace("-", " ").strip()


def _names_match(a: str, b: str) -> bool:
    """Return True if two fighter names refer to the same person, accounting for
    punctuation differences and known aliases."""
    if _normalize_name(a) == _normalize_name(b):
        return True
    # Also check via _NAME_FIXES: resolve both to canonical form and compare
    a_canon = _normalize_name(_NAME_FIXES.get(a.lower(), a))
    b_canon = _normalize_name(_NAME_FIXES.get(b.lower(), b))
    return a_canon == b_canon


def _fight_lookup_key_candidates(fighter1: str, fighter2: str) -> list[frozenset[str]]:
    """Return raw and alias-canonical lookup keys for a fighter pair."""
    raw_key = frozenset([fighter1.lower(), fighter2.lower()])
    canon1 = _NAME_FIXES.get(fighter1.lower(), fighter1).lower()
    canon2 = _NAME_FIXES.get(fighter2.lower(), fighter2).lower()
    canonical_key = frozenset([canon1, canon2])
    return [raw_key] if canonical_key == raw_key else [raw_key, canonical_key]


def _resolve_fighter_for_backtest(session, name: str):
    """Resolve a fighter name to a DB Fighter, preferring the one with most fights."""
    from sqlalchemy import or_
    from database.schema import Fighter, Fight
    # Apply known aliases before DB lookup
    name = _NAME_FIXES.get(name.lower(), name)
    rows = session.query(Fighter).filter(Fighter.name.ilike(f"%{name}%")).all()
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    # Pick the fighter with the most DB fights to avoid matching wrong namesakes
    scored = sorted(
        rows,
        key=lambda f: session.query(Fight).filter(
            or_(Fight.fighter_1_id == f.id, Fight.fighter_2_id == f.id)
        ).count(),
        reverse=True,
    )
    return scored[0]


def run_prediction(
    fighter1: str,
    fighter2: str,
    model_name: str,
    as_of_date: str | None = None,
    market_prob_f1: float | None = None,
    session=None,
    underdog_blend: bool = True,
) -> tuple[float | None, float | None]:
    """
    Run a symmetric mar_4_v2 prediction.  When underdog_blend=True, also applies
    the underdog_v1 blend whenever either fighter is a market underdog
    (implied prob < UD_THRESHOLD), mirroring predict_service._score_row.

    Returns (prob_f1, prob_f2).
    """
    try:
        f1_obj = _resolve_fighter_for_backtest(session, fighter1)
        f2_obj = _resolve_fighter_for_backtest(session, fighter2)
        if not f1_obj or not f2_obj:
            missing = [n for n, o in [(fighter1, f1_obj), (fighter2, f2_obj)] if not o]
            print(f"  Fighter not found: {', '.join(missing)}")
            return None, None

        # Symmetric general model: average both orderings
        p_f1_fwd = _predict_one_direction(session, f1_obj.id, f2_obj.id, model_name, as_of_date)
        p_f1_rev = _predict_one_direction(session, f2_obj.id, f1_obj.id, model_name, as_of_date)
        gen_prob_f1 = 0.5 * (p_f1_fwd + (1.0 - p_f1_rev))

        if underdog_blend and market_prob_f1 is not None:
            if market_prob_f1 < UD_THRESHOLD:
                # f1 is the market underdog → blend underdog_v1 from f1's perspective
                p_ud = _predict_one_direction(session, f1_obj.id, f2_obj.id, "underdog_v1", as_of_date)
                gen_prob_f1 = BLEND_WEIGHT * p_ud + (1 - BLEND_WEIGHT) * gen_prob_f1
            elif (1.0 - market_prob_f1) < UD_THRESHOLD:
                # f2 is the market underdog → blend underdog_v1 from f2's perspective
                p_ud_f2 = _predict_one_direction(session, f2_obj.id, f1_obj.id, "underdog_v1", as_of_date)
                gen_prob_f2 = BLEND_WEIGHT * p_ud_f2 + (1 - BLEND_WEIGHT) * (1.0 - gen_prob_f1)
                gen_prob_f1 = 1.0 - gen_prob_f2

        return gen_prob_f1, 1.0 - gen_prob_f1

    except Exception as e:
        print(f"  Error predicting {fighter1} vs {fighter2}: {e}")
        return None, None


def odds_to_probability(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


def calculate_ev(model_prob, american_odds):
    """Calculate expected value for a bet."""
    implied_prob = odds_to_probability(american_odds)

    if american_odds > 0:
        profit = american_odds / 100
    else:
        profit = 100 / -american_odds

    ev = (model_prob * profit) - ((1 - model_prob) * 1)
    return ev, implied_prob


def pnl_for_bet(won: bool, american_odds: int) -> float:
    """Return profit/loss per 1 unit staked. Won=True returns the payout, won=False returns -1."""
    if not won:
        return -1.0
    if american_odds > 0:
        return american_odds / 100
    else:
        return 100 / -american_odds


def should_bet(pick_odds: int, pick_prob: float, pick_ev: float | None, cfg: dict) -> tuple[bool, str]:
    """
    Decide whether to place a bet based on config constraints.
    Returns (should_bet: bool, reason: str).

    Edge thresholds (edge_min, edge_underdog) are compared against the
    probability edge = model_prob − market_implied_prob, matching the site's
    "Edge: +X%" display convention.
    """
    is_favorite = pick_odds < 0
    edge_min = cfg.get("edge_min", 0.05)
    edge_underdog = cfg.get("edge_underdog", edge_min)
    conf_fav = cfg.get("confidence_favorite", 0.50)
    conf_ud = cfg.get("confidence_underdog", 0.50)
    fav_cap = cfg.get("favorite_odds_cap", None)
    ud_cap = cfg.get("underdog_odds_cap", None)

    market_prob = odds_to_probability(pick_odds)
    prob_edge = pick_prob - market_prob

    if is_favorite:
        # Favorite cap: skip if odds are AT or beyond the cap (matches site's <= check)
        if fav_cap is not None and pick_odds <= fav_cap:
            return False, f"favorite cap ({pick_odds} <= {fav_cap})"
        # Confidence check
        if pick_prob < conf_fav:
            return False, f"favorite confidence ({pick_prob:.1%} < {conf_fav:.1%})"
        # Edge check (probability edge)
        if edge_min != 0 and prob_edge < edge_min:
            return False, f"edge ({prob_edge:.1%} < {edge_min:.1%})"
    else:
        # Underdog cap: skip if odds are AT or beyond the cap (matches site's >= check)
        if ud_cap is not None and pick_odds >= ud_cap:
            return False, f"underdog cap ({pick_odds} >= {ud_cap})"
        # Underdog confidence check
        if pick_prob < conf_ud:
            return False, f"underdog confidence ({pick_prob:.1%} < {conf_ud:.1%})"
        # Underdog edge check (probability edge)
        if prob_edge < edge_underdog:
            return False, f"underdog edge ({prob_edge:.1%} < {edge_underdog:.1%})"

    return True, "passed"


def load_fight_dates(session) -> dict:
    """
    Load fight dates per fighter from DB.
    Returns {fighter_name_lower: [date_str, ...]} so callers can compute
    point-in-time fight counts (fights before a given date) without leakage.
    Also adds _NAME_FIXES aliases so odds-CSV names resolve correctly.
    """
    from database.schema import Fight, Fighter, Event
    rows = (
        session.query(Fighter.name, Event.date)
        .select_from(Fighter)
        .join(Fight, (Fight.fighter_1_id == Fighter.id) | (Fight.fighter_2_id == Fighter.id))
        .join(Event, Fight.event_id == Event.id)
        .filter(Event.date.isnot(None))
        .all()
    )
    result: dict[str, list[str]] = {}
    for name, date_str in rows:
        key = name.lower()
        result.setdefault(key, []).append(str(date_str))

    # Add reverse aliases: CSV name → same list as DB name so the lookup works
    # even when the odds source uses a different spelling than the DB.
    for csv_name, db_name in _NAME_FIXES.items():
        db_key = db_name.lower()
        if db_key in result and csv_name not in result:
            result[csv_name] = result[db_key]

    return result


def _fights_before(fight_dates: dict, fighter_name: str, as_of: str) -> int:
    """Count how many DB fights a fighter had strictly before *as_of* (YYYY-MM-DD).

    Tries the raw name first, then the _NAME_FIXES DB canonical name, then a
    partial-match fallback so minor spelling differences don't silently return 0
    and trigger the min_fights filter incorrectly.
    """
    name_lower = fighter_name.lower()
    dates = fight_dates.get(name_lower)

    if dates is None:
        # Try the canonical DB name via _NAME_FIXES
        db_name = _NAME_FIXES.get(name_lower, '').lower()
        if db_name:
            dates = fight_dates.get(db_name)

    if dates is None:
        # Partial-match fallback: DB name contains or is contained by the CSV name
        for key, val in fight_dates.items():
            if name_lower in key or key in name_lower:
                dates = val
                break

    if not dates:
        # Name truly not found — return a sentinel that won't filter incorrectly.
        # Return None so callers can distinguish "not found" from "0 fights".
        return None  # type: ignore[return-value]

    cutoff = pd.to_datetime(as_of)
    return sum(1 for d in dates if pd.to_datetime(d, errors='coerce') < cutoff)


def load_weight_classes(session) -> dict:
    """Load weight_class per fight from DB. Keyed by frozenset of lowercased fighter names."""
    from database.schema import Fight, Fighter, Event
    from sqlalchemy.orm import aliased
    F1 = aliased(Fighter)
    F2 = aliased(Fighter)
    rows = (
        session.query(F1.name, F2.name, Fight.weight_class)
        .select_from(Fight)
        .join(F1, Fight.fighter_1_id == F1.id)
        .join(F2, Fight.fighter_2_id == F2.id)
        .filter(Fight.weight_class.isnot(None))
        .all()
    )
    return {frozenset([n1.lower(), n2.lower()]): wc for n1, n2, wc in rows}


def apply_strategy(results_df: pd.DataFrame, cfg: dict, fight_dates: dict, weight_classes: dict = None) -> pd.DataFrame:
    """
    Re-apply betting strategy from config to an existing results DataFrame.
    Adds/overwrites 'bet' and 'skip_reason' columns. Does NOT re-run predictions.
    """
    df = results_df.copy()
    min_fights = cfg.get("min_fights", 0)
    bet_female = cfg.get("female", True)

    bets = []
    reasons = []

    for _, row in df.iterrows():
        pick_odds = row.get('pick_odds')
        pick_prob = row.get('pick_prob')
        pick = row.get('pick')
        f1 = row.get('fighter1', '')
        f2 = row.get('fighter2', '')

        # Config-driven filters are evaluated first so that changing the config
        # in --results mode actually affects which fights are included/excluded,
        # even if those fights were previously skipped for a different reason.

        # Check female fights — prefer pre-computed 'female' column, fall back to DB dict
        if not bet_female:
            if 'female' in row and pd.notna(row['female']):
                is_female = bool(row['female'])
            elif weight_classes:
                key = frozenset([f1.lower(), f2.lower()])
                wc = weight_classes.get(key, '')
                is_female = bool(wc and wc.lower().startswith("women"))
            else:
                is_female = False
            if is_female:
                bets.append(False)
                reasons.append('female')
                continue

        # Check min_fights (point-in-time: only count fights before this fight's date)
        if min_fights > 0 and fight_dates:
            row_date = str(row.get('date', ''))
            f1_count = _fights_before(fight_dates, f1, row_date) if row_date else None
            f2_count = _fights_before(fight_dates, f2, row_date) if row_date else None
            print(f1,f1_count)
            print(f2,f2_count)
            # None means the fighter wasn't found in the DB at all — treat as 0 fights.
            f1_fail = (f1_count is None) or (f1_count < min_fights)
            f2_fail = (f2_count is None) or (f2_count < min_fights)
            if f1_fail or f2_fail:
                bets.append(False)
                reasons.append('min_fights')
                continue

        # Now bail on rows where prediction genuinely failed (no pick produced)
        if row.get('error', False) and pd.isna(row.get('pick')):
            bets.append(False)
            reasons.append(row.get('skip_reason') or 'error')
            continue

        # Derive pick_ev from ev1/ev2 based on which fighter was picked
        pick_ev = None
        if pick_odds is not None and pick_prob is not None:
            if pick == f1:
                pick_ev = row.get('ev1')
            else:
                pick_ev = row.get('ev2')

        if pick_odds is not None and pick_ev is not None:
            bet, reason = should_bet(int(pick_odds), pick_prob, pick_ev, cfg)
        else:
            bet = False
            reason = "missing odds"

        bets.append(bet)
        reasons.append(None if bet else reason)

    df['bet'] = bets
    df['skip_reason'] = reasons
    return df


def print_summary(results_df: pd.DataFrame, cfg: dict, quiet: bool):
    """Print backtest summary for a results DataFrame that has 'bet' and 'skip_reason' columns."""
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    successful = results_df[~results_df['error'].fillna(True)].copy()
    failed = results_df[results_df['error'].fillna(True)].copy()

    print(f"\nTotal fights:       {len(results_df)}")
    print(f"Predicted:          {len(successful)}")
    print(f"Prediction failed:  {len(failed)}")
    if len(failed) > 0 and 'skip_reason' in failed.columns:
        fail_reasons = failed['skip_reason'].dropna().value_counts()
        for reason, count in fail_reasons.items():
            print(f"  {reason}: {count}")

    # --- CONFIG-DRIVEN BETTING STRATEGY ---
    bet_df = successful[successful['bet']].copy()
    no_bet_df = successful[~successful['bet']].copy()

    print(f"\n--- BETTING STRATEGY ---")
    print(f"    Bets placed:    {len(bet_df)}/{len(successful)}")
    print(f"    Bets skipped:   {len(no_bet_df)}/{len(successful)}")

    if len(no_bet_df) > 0 and 'skip_reason' in no_bet_df.columns:
        skip_counts = no_bet_df['skip_reason'].dropna().value_counts()
        if len(skip_counts) > 0:
            print(f"    Skip reasons:")
            for reason, count in skip_counts.items():
                print(f"      {reason}: {count}")

    # Actual results for placed bets
    bet_with_results = bet_df.dropna(subset=['pick_correct']).copy() if 'pick_correct' in bet_df.columns else pd.DataFrame()
    if len(bet_with_results) > 0:
        n_correct = bet_with_results['pick_correct'].sum()
        n_total = len(bet_with_results)
        win_pct = n_correct / n_total * 100
        total_pnl = bet_with_results['actual_pnl'].sum()
        realized_roi = total_pnl / n_total * 100

        print(f"\n--- ACTUAL RESULTS (placed bets only) ---")
        print(f"    Bets with known winner: {n_total}")
        print(f"    Correct: {int(n_correct)}/{n_total} ({win_pct:.1f}%)")
        print(f"    Realized P&L: {total_pnl:+.2f} units (1 unit per bet)")
        print(f"    Realized ROI: {realized_roi:+.1f}%")

        # Breakdown: favorites vs underdogs among placed bets
        fav_bets = bet_with_results[bet_with_results['pick_odds'] < 0]
        ud_bets = bet_with_results[bet_with_results['pick_odds'] > 0]
        if len(fav_bets) > 0:
            fav_correct = fav_bets['pick_correct'].sum()
            fav_pnl = fav_bets['actual_pnl'].sum()
            fav_roi = fav_pnl / len(fav_bets) * 100
            print(f"\n    Favorites: {int(fav_correct)}/{len(fav_bets)} ({fav_correct/len(fav_bets)*100:.1f}%), P&L {fav_pnl:+.2f}, ROI {fav_roi:+.1f}%")
        if len(ud_bets) > 0:
            ud_correct = ud_bets['pick_correct'].sum()
            ud_pnl = ud_bets['actual_pnl'].sum()
            ud_roi = ud_pnl / len(ud_bets) * 100
            print(f"    Underdogs: {int(ud_correct)}/{len(ud_bets)} ({ud_correct/len(ud_bets)*100:.1f}%), P&L {ud_pnl:+.2f}, ROI {ud_roi:+.1f}%")
    else:
        print(f"\n--- ACTUAL RESULTS ---")
        print("    No bets placed, or no winners found in DB.")

    # --- ALL PICKS (for comparison, no strategy filter) ---
    if 'pick_correct' in successful.columns:
        all_with_results = successful.dropna(subset=['pick_correct']).copy()
        if len(all_with_results) > 0:
            all_correct = all_with_results['pick_correct'].sum()
            all_pnl = all_with_results['actual_pnl'].sum()
            all_roi = all_pnl / len(all_with_results) * 100
            print(f"\n--- ALL PICKS (no strategy filter, baseline) ---")
            print(f"    Correct: {int(all_correct)}/{len(all_with_results)} ({all_correct/len(all_with_results)*100:.1f}%)")
            print(f"    P&L: {all_pnl:+.2f} units, ROI: {all_roi:+.1f}%")

    # --- INDIVIDUAL BET DETAILS ---
    if len(bet_with_results) > 0 and not quiet:
        print(f"\n--- PLACED BET DETAILS ---")
        for _, r in bet_with_results.iterrows():
            outcome = "WON " if r['pick_correct'] else "LOST"
            pnl_str = f"+{r['actual_pnl']:.2f}" if r['actual_pnl'] >= 0 else f"{r['actual_pnl']:.2f}"
            pick_ev = r.get('ev1') if r['pick'] == r['fighter1'] else r.get('ev2')
            ev_str = f"{pick_ev:+.2f}" if pd.notna(pick_ev) else "N/A"
            pick_odds = int(r['pick_odds']) if pd.notna(r['pick_odds']) else 0
            print(f"  [{r['date']}] {r['pick']:25s} @ {pick_odds:+5d}  "
                  f"prob={r['pick_prob']:.1%}  ev={ev_str}  "
                  f"{outcome} ({pnl_str})  vs {r['fighter2'] if r['pick']==r['fighter1'] else r['fighter1']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Backtest model predictions vs odds")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to backtest_config.json")
    parser.add_argument("--results", type=str, default=None, help="Re-use an existing backtest results CSV (skip predictions)")
    parser.add_argument("--odds", type=str, default=str(DEFAULT_ODDS_CSV), help="Path to odds CSV")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--quiet", action="store_true", help="Only print summary, not per-fight")
    parser.add_argument("--cutoff", type=str, default=None, help="Override cutoff date from config (YYYY-MM-DD)")
    parser.add_argument("--results-out", type=str, default=None, help="Output path for results CSV (default: inferred from odds year, else backtest/backtest_results.csv)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / config_path
    cfg = load_config(config_path)

    model_name = args.model or cfg.get("model", "mar_4_v2")
    cutoff = args.cutoff or cfg.get("cutoff_date", "2026-03-01")

    print(f"Config: {config_path}")
    print(f"Model:  {model_name}")
    print(f"Cutoff: {cutoff}")
    print(f"Strategy: edge_min={cfg.get('edge_min', 0.05)}, edge_underdog={cfg.get('edge_underdog', 0.03)}, "
          f"conf_fav={cfg.get('confidence_favorite', 0.55)}, conf_ud={cfg.get('confidence_underdog', 0.50)}, "
          f"fav_cap={cfg.get('favorite_odds_cap', 'none')}, ud_cap={cfg.get('underdog_odds_cap', 'none')}, "
          f"min_fights={cfg.get('min_fights', 0)}, female={cfg.get('female', True)}")

    # Open one shared DB session for the full run
    db = DatabaseManager()
    pred_session = db.get_session()

    # Load fight dates and weight classes from DB (needed even in --results mode)
    try:
        fight_dates = load_fight_dates(pred_session)
        weight_classes = load_weight_classes(pred_session)
    except Exception:
        fight_dates = {}
        weight_classes = {}

    # ── Shortcut: re-use existing results file ──────────────────────
    if args.results:
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return

        results_df = pd.read_csv(results_path)
        # Apply cutoff filter
        if 'date' in results_df.columns:
            results_df['date'] = pd.to_datetime(results_df['date'], errors='coerce')
            cutoff_date = pd.to_datetime(cutoff)
            results_df = results_df[results_df['date'] < cutoff_date].copy()
            results_df['date'] = results_df['date'].dt.strftime('%Y-%m-%d')

        print(f"\nLoaded {len(results_df)} results from {results_path} (skipping predictions)")
        print("=" * 80)

        results_df = apply_strategy(results_df, cfg, fight_dates, weight_classes)
        print_summary(results_df, cfg, args.quiet)
        pred_session.close()
        return

    # ── Full run: load odds, run predictions ────────────────────────
    csv_path = Path(args.odds)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        print(f"Odds file not found: {csv_path}")
        print("Export from DB:  python scripts/export_odds_from_db.py --year 2025 -o backtest/odds/db_2025.csv")
        return

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        print("CSV must have columns: date, fighter1, fighter2, fighter1_odds, fighter2_odds")
        return

    print(f"\nLoaded {len(df)} fights from {csv_path}")
    print("=" * 80)

    # Load fight results from DB
    try:
        winner_lookup = db.get_fight_results_lookup()
        print(f"Loaded {len(winner_lookup)} fight results from DB")
    except Exception as e:
        print(f"Warning: could not load fight results from DB ({e})")
        winner_lookup = {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    cutoff_date = pd.to_datetime(cutoff)
    past_fights = df[df["date"] < cutoff_date].copy()

    print(f"Fights before {cutoff}: {len(past_fights)}")
    print("=" * 80)

    results = []

    for idx, row in past_fights.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        provenance = odds_provenance(row, odds_path=csv_path, source_line=int(idx) + 2)
        # Use the fight date as the feature cutoff.  get_fight_history uses a
        # strict < comparison, and MatchupFeatureExtractor additionally excludes
        # the predicted bout by fighter-pair identity within a few days of the
        # anchor — so the fight can't leak even when the odds-CSV date is a day
        # off from the DB event date (a common cross-source disagreement).
        as_of_date = date
        f1 = row["fighter1"]
        f2 = row["fighter2"]
        odds1 = int(row["fighter1_odds"]) if pd.notna(row.get("fighter1_odds")) else None
        odds2 = int(row["fighter2_odds"]) if pd.notna(row.get("fighter2_odds")) else None

        if not args.quiet:
            print(f"\n[{date}] {f1} vs {f2}")
            print(f"  Odds: {f1} {odds1}, {f2} {odds2}")

        # No pre-prediction filtering by female or min_fights.
        # All fights get a prediction attempt so the CSV is a complete record.
        # Strategy filters (female, min_fights, edge, confidence) are applied
        # by apply_strategy so they can be toggled freely in --results mode.
        market_prob_f1 = odds_to_probability(odds1) if odds1 is not None else None
        prob_f1, prob_f2 = run_prediction(
            f1, f2, model_name,
            as_of_date=as_of_date,
            market_prob_f1=market_prob_f1,
            session=pred_session,
            underdog_blend=False,
        )

        if prob_f1 is None:
            if not args.quiet:
                print("  Prediction failed")
            results.append({
                'date': date, 'main_fight_id': None, 'fighter1': f1, 'fighter2': f2,
                'odds1': odds1, 'odds2': odds2,
                'prob1': None, 'prob2': None, 'pick': None,
                'pick_odds': None, 'pick_prob': None,
                'ev1': None, 'ev2': None,
                'winner': None, 'pick_correct': None, 'actual_pnl': None,
                'bet': False, 'skip_reason': 'prediction_failed',
                'error': True,
                **provenance,
            })
            continue

        if not args.quiet:
            print(f"{as_of_date}  Model: {f1} {prob_f1*100:.1f}%, {f2} {prob_f2*100:.1f}%")

        ev1, imp1 = calculate_ev(prob_f1, odds1) if odds1 is not None else (None, None)
        ev2, imp2 = calculate_ev(prob_f2, odds2) if odds2 is not None else (None, None)

        if not args.quiet and ev1 is not None:
            print(f"  EV: {f1} ${ev1:.2f} (implied {imp1*100:.1f}%), {f2} ${ev2:.2f} (implied {imp2*100:.1f}%)")

        if prob_f1 > prob_f2:
            pick = f1
            pick_odds = odds1
            pick_prob = prob_f1
            pick_ev = ev1
        else:
            pick = f2
            pick_odds = odds2
            pick_prob = prob_f2
            pick_ev = ev2

        # Look up actual winner — prefer DB (exact rematch-aware), fall back to CSV winner column
        fight_list = []
        for key in _fight_lookup_key_candidates(f1, f2):
            fight_list = winner_lookup.get(key, [])
            if fight_list:
                break
        if len(fight_list) == 1:
            fight_result = fight_list[0]
        elif len(fight_list) > 1:
            fight_result = next(
                (r for r in fight_list if r.get("date") and str(r["date"])[:10] == date),
                fight_list[-1],  # fallback: most recently inserted row
            )
        else:
            fight_result = None
        main_fight_id = fight_result.get("fight_id") if fight_result else None
        winner = fight_result["winner"] if fight_result else None

        # Fallback: use winner column from the odds CSV (already resolved by rebuild script)
        if winner is None and "winner" in row and pd.notna(row["winner"]):
            winner = str(row["winner"])

        if winner and pick_odds is not None:
            pick_correct = _names_match(pick, winner)
            actual_pnl = pnl_for_bet(pick_correct, pick_odds)
        else:
            pick_correct = None
            actual_pnl = None

        # Apply betting strategy from config
        if pick_odds is not None and pick_ev is not None:
            bet, bet_reason = should_bet(pick_odds, pick_prob, pick_ev, cfg)
        else:
            bet = False
            bet_reason = "missing odds"

        if not args.quiet:
            if winner:
                outcome = "CORRECT" if pick_correct else "WRONG"
                bet_str = "BET" if bet else "NO BET"
                print(f"  Winner: {winner}  |  Pick: {outcome}  |  {bet_str}" + (f" ({bet_reason})" if not bet else ""))
            elif not bet:
                print(f"  NO BET: {bet_reason}")

        # Determine if this is a women's fight
        wc_key = frozenset([f1.lower(), f2.lower()])
        wc = weight_classes.get(wc_key, '') if weight_classes else ''
        is_female = bool(wc and wc.lower().startswith('women'))

        results.append({
            'date': date, 'main_fight_id': main_fight_id, 'fighter1': f1, 'fighter2': f2,
            'odds1': odds1, 'odds2': odds2,
            'prob1': prob_f1, 'prob2': prob_f2,
            'pick': pick, 'pick_odds': pick_odds, 'pick_prob': pick_prob,
            'ev1': ev1, 'ev2': ev2,
            'winner': winner, 'pick_correct': pick_correct, 'actual_pnl': actual_pnl,
            'bet': bet, 'skip_reason': None if bet else bet_reason,
            'error': False, 'female': is_female,
            **provenance,
        })

    pred_session.close()

    results_df = pd.DataFrame(results)

    # Re-apply the full strategy (including min_fights and female filters) so
    # the saved CSV and summary are consistent with --results mode.
    results_df = apply_strategy(results_df, cfg, fight_dates, weight_classes)

    if args.results_out:
        out_path = Path(args.results_out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = default_results_path_for_odds(csv_path)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    print_summary(results_df, cfg, args.quiet)


if __name__ == "__main__":
    main()
