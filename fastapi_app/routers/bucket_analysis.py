"""Bucket-analysis API — serves pre-computed backtest stats as JSON.

Mirrors the CLI output of ``backtest/bucket_analysis.py`` for the two
canonical result sets (2025 completed season + 2026 in-progress).
"""

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

from fastapi import APIRouter

# Make the repo root importable so we can reuse bucket_analysis helpers
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backtest.bucket_analysis import (
    BUCKET_RANGES,
    EDGE_TIERS,
    _get_multiplier,
    _stats,
    assign_bucket,
    assign_edge_tier,
    market_implied_prob,
    parse_bets_txt,
    parse_csv,
)

router = APIRouter(tags=["bucket-analysis"])

# ── File paths ──────────────────────────────────────────────────────────────

_BACKTEST_DIR = _REPO_ROOT / "backtest"
_CONFIG_PATH = _REPO_ROOT / "config" / "betting_config.json"

DATASETS = {
    "2025": {
        "results": _BACKTEST_DIR / "backtest_2025_results.csv",
        "bets": _BACKTEST_DIR / "bets_2025.txt",
    },
    "2026": {
        "results": _BACKTEST_DIR / "backtest_2026_results.csv",
        "bets": _BACKTEST_DIR / "bets.txt",
    },
}

# ── Weight-class lookup from DB ─────────────────────────────────────────────

_DB_PATH = _REPO_ROOT / "data" / "ufc_database.db"


def _normalize(name: str) -> str:
    return re.sub(r"['\.\-]", "", name).lower().strip()


def _build_weight_class_lookup() -> dict[tuple[str, str], str]:
    """Build (normalized_f1, normalized_f2) -> weight_class from the fights DB.

    Primary: exact fight match. Fallback: fighter's most recent weight class.
    """
    if not _DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(_DB_PATH))
    c = conn.cursor()

    # Primary: exact fight pairing -> weight class
    c.execute("""
        SELECT fi1.name, fi2.name, f.weight_class
        FROM fights f
        JOIN fighters fi1 ON f.fighter_1_id = fi1.id
        JOIN fighters fi2 ON f.fighter_2_id = fi2.id
        WHERE f.weight_class IS NOT NULL
    """)
    lookup = {}
    for f1, f2, wc in c.fetchall():
        nf1, nf2 = _normalize(f1), _normalize(f2)
        lookup[(nf1, nf2)] = wc
        lookup[(nf2, nf1)] = wc

    # Fallback: each fighter's most recent weight class
    c.execute("""
        SELECT fi.name, f.weight_class
        FROM fights f
        JOIN events e ON f.event_id = e.id
        JOIN fighters fi ON fi.id IN (f.fighter_1_id, f.fighter_2_id)
        WHERE f.weight_class IS NOT NULL
          AND f.weight_class NOT IN ('Catch Weight', 'Open Weight')
        ORDER BY e.date DESC
    """)
    fighter_wc: dict[str, str] = {}
    for name, wc in c.fetchall():
        n = _normalize(name)
        if n not in fighter_wc:
            fighter_wc[n] = wc

    conn.close()
    return lookup, fighter_wc


_WC_LOOKUP: tuple[dict, dict] | None = None


def _get_wc_lookup():
    global _WC_LOOKUP
    if _WC_LOOKUP is None:
        _WC_LOOKUP = _build_weight_class_lookup()
    return _WC_LOOKUP


def _parse_csv_with_weight_class(results_path: Path, bets_filter):
    """Parse CSV rows and attach weight_class from DB lookup."""
    fight_lookup, fighter_wc = _get_wc_lookup()
    rows_with_wc = []

    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("pick_prob"):
                continue
            if bets_filter is not None:
                key = (row.get("date", ""), _normalize(row.get("pick", "")))
                if key not in bets_filter:
                    continue

            f1 = _normalize(row.get("fighter1", ""))
            f2 = _normalize(row.get("fighter2", ""))

            # Try exact fight pairing first, then fallback to fighter's recent class
            wc = fight_lookup.get((f1, f2))
            if not wc:
                wc = fighter_wc.get(f1) or fighter_wc.get(f2) or "Unknown"

            rows_with_wc.append((row, wc))

    return rows_with_wc


def _bucket_prob_range(bucket_key: int) -> str:
    lo_odds, hi_odds = BUCKET_RANGES[bucket_key]
    lo_prob = market_implied_prob(max(lo_odds, -2000))
    hi_prob = market_implied_prob(min(hi_odds, 2000))
    p_hi = round(lo_prob * 100)
    p_lo = round(hi_prob * 100)
    return f"{p_lo}%–{p_hi}%"


def _stats_dict(entries):
    n, w, l, wr, profit, roi, avg_edge, avg_conf = _stats(entries)
    return {
        "n": n, "w": w, "l": l,
        "win_rate": round(wr * 100, 1),
        "profit": round(profit, 2),
        "roi": round(roi, 1),
        "avg_edge": round(avg_edge * 100, 1),
        "avg_conf": round(avg_conf * 100, 1),
    }


def _gender_split(entries):
    """Return ALL / M / F stats dicts."""
    male = [e for e in entries if not e.female]
    female = [e for e in entries if e.female]
    result = {"ALL": _stats_dict(entries)}
    if male:
        result["M"] = _stats_dict(male)
    if female:
        result["F"] = _stats_dict(female)
    return result


# ── Builders ────────────────────────────────────────────────────────────────

EXCLUDED_BUCKETS = {-400, 400}


def _build_odds_buckets(rows):
    buckets = {k: [] for k in BUCKET_RANGES if k not in EXCLUDED_BUCKETS}
    for r in rows:
        key = assign_bucket(r.pick_odds)
        if key is not None and key not in EXCLUDED_BUCKETS:
            buckets[key].append(r)

    result = []
    best_roi, best_label = None, None
    for key in sorted(buckets.keys()):
        entries = buckets[key]
        label = f"{key:+d} ({_bucket_prob_range(key)})"
        stats = _gender_split(entries)
        result.append({"label": label, "key": key, **stats})
        if entries:
            roi = stats["ALL"]["roi"]
            if best_roi is None or roi > best_roi:
                best_roi = roi
                best_label = label

    all_entries = rows
    totals = _gender_split(all_entries)
    return {"buckets": result, "totals": totals, "best_bucket": best_label}


def _build_edge_tiers(rows):
    tiers = {label: [] for label in EDGE_TIERS}
    for r in rows:
        tier = assign_edge_tier(r.edge)
        if tier is not None:
            tiers[tier].append(r)

    result = []
    for label in EDGE_TIERS:
        entries = tiers[label]
        stats = _gender_split(entries)
        result.append({"label": label, **stats})

    totals = _gender_split(rows)
    return {"tiers": result, "totals": totals}


def _build_weighted_roi(rows, config: dict):
    base_unit = config.get("betting", {}).get("base_unit", 100)
    cfg_buckets = config.get("edge_buckets", [])

    tiers = []
    total_staked = total_profit = flat_staked = flat_profit = 0.0
    total_n = total_w = 0

    for b in cfg_buckets:
        lo, hi = b["min_edge"], b["max_edge"]
        action = b.get("action")
        mult = b.get("multiplier")
        tier_rows = [r for r in rows if lo <= r.edge < hi]
        n = len(tier_rows)
        w = sum(1 for r in tier_rows if r.correct)
        l = n - w
        wr = (w / n * 100) if n else 0
        avg_edge = (sum(r.edge for r in tier_rows) / n * 100) if n else 0

        if action == "skip":
            label = f"{lo*100:.0f}–{hi*100:.0f}% (skip)"
            tiers.append({
                "label": label, "mult": "skip", "n": n, "w": w, "l": l,
                "win_rate": round(wr, 1), "staked": None, "profit": None,
                "roi": None, "avg_edge": round(avg_edge, 1),
            })
            continue

        label = f"{lo*100:.0f}–{hi*100:.0f}% ({mult}x)"
        staked = n * base_unit * mult
        profit = sum(r.pnl * base_unit * mult for r in tier_rows)
        roi = (profit / staked * 100) if staked else 0

        total_staked += staked
        total_profit += profit
        total_n += n
        total_w += w
        flat_staked += n * base_unit
        flat_profit += sum(r.pnl * base_unit for r in tier_rows)

        tiers.append({
            "label": label, "mult": f"{mult}x", "n": n, "w": w, "l": l,
            "win_rate": round(wr, 1),
            "staked": round(staked),
            "profit": round(profit, 2),
            "roi": round(roi, 1),
            "avg_edge": round(avg_edge, 1),
        })

    total_roi = (total_profit / total_staked * 100) if total_staked else 0
    flat_roi = (flat_profit / flat_staked * 100) if flat_staked else 0
    total_wr = (total_w / total_n * 100) if total_n else 0

    return {
        "base_unit": base_unit,
        "tiers": tiers,
        "weighted": {
            "n": total_n, "w": total_w, "l": total_n - total_w,
            "win_rate": round(total_wr, 1),
            "staked": round(total_staked),
            "profit": round(total_profit, 2),
            "roi": round(total_roi, 1),
        },
        "flat": {
            "n": total_n, "w": total_w, "l": total_n - total_w,
            "win_rate": round(total_wr, 1),
            "staked": round(flat_staked),
            "profit": round(flat_profit, 2),
            "roi": round(flat_roi, 1),
        },
        "lift_pp": round(total_roi - flat_roi, 1),
    }


def _build_weight_class_performance(rows_with_wc):
    """Group FightData by weight class and compute stats."""
    from collections import defaultdict
    from backtest.bucket_analysis import FightData, market_implied_prob as mip

    by_wc = defaultdict(list)
    for raw_row, wc in rows_with_wc:
        pick_prob = float(raw_row["pick_prob"])
        pick_odds = float(raw_row["pick_odds"]) if raw_row.get("pick_odds") else 0.0
        mkt_prob = mip(pick_odds)
        edge = pick_prob - mkt_prob
        pnl = float(raw_row["actual_pnl"]) if raw_row.get("actual_pnl") else 0.0
        correct = raw_row.get("pick_correct") == "True"

        by_wc[wc].append(FightData(
            correct=correct,
            female=raw_row.get("female") == "True",
            edge=edge,
            pnl=pnl,
            pick_odds=pick_odds,
            pick_prob=pick_prob,
            bet=raw_row.get("bet") == "True",
            skip_reason=raw_row.get("skip_reason", "").strip(),
        ))

    result = []
    for wc in sorted(by_wc.keys()):
        entries = by_wc[wc]
        stats = _stats_dict(entries)
        result.append({"weight_class": wc, **stats})

    # Sort by number of bets descending
    result.sort(key=lambda x: x["n"], reverse=True)
    return result


def _build_fights_table(results_path: Path, bets_filter: set | None):
    """Build a minimal fight-by-fight table for display. Includes all fights."""
    fight_lookup, fighter_wc = _get_wc_lookup()
    fights = []

    # Map skip reasons to short codes
    skip_codes = {
        "favorite confidence": "F1",
        "favorite cap": "F2",
        "underdog confidence": "U1",
        "underdog edge": "U2",
        "underdog cap": "U3",
        "min_fights": "D1",
        "prediction_failed": "ERR",
    }

    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("pick_prob"):
                continue

            pick_prob = float(row["pick_prob"])
            pick_odds = float(row["pick_odds"]) if row.get("pick_odds") else 0
            pnl_raw = row.get("actual_pnl", "").strip()
            pnl = round(float(pnl_raw), 2) if pnl_raw else 0.0
            correct = row.get("pick_correct") == "True"
            skip_reason = row.get("skip_reason", "").strip()

            # Use bets.txt as source of truth for actual bets placed
            actually_bet = False
            if bets_filter is not None:
                key = (row.get("date", ""), _normalize(row.get("pick", "")))
                actually_bet = key in bets_filter

            # Map skip reason to code
            skip_code = ""
            if not actually_bet:
                if skip_reason:
                    for prefix, code in skip_codes.items():
                        if skip_reason.startswith(prefix):
                            skip_code = code
                            break
                    if not skip_code:
                        skip_code = "?"
                else:
                    # CSV said bet=True but not in bets.txt — infer reason from data
                    mkt_prob = market_implied_prob(pick_odds)
                    edge = pick_prob - mkt_prob
                    cfg = json.loads(_CONFIG_PATH.read_text())
                    filters = cfg.get("filters", {})
                    is_fav = pick_odds < 0
                    fav_conf = filters.get("favorite_confidence_min")
                    if is_fav and fav_conf is not None and pick_prob < fav_conf:
                        skip_code = "F1"
                    elif is_fav and pick_odds <= -abs(filters.get("favorite_odds_cap", 300)):
                        skip_code = "F2"
                    elif not is_fav and pick_prob < filters.get("underdog_confidence_min", 0.53):
                        skip_code = "U1"
                    elif is_fav and edge < filters.get("edge_min", 0.05):
                        skip_code = "U2"
                    elif not is_fav and edge < filters.get("underdog_edge_min", filters.get("edge_min", 0.05)):
                        skip_code = "U2"
                    else:
                        skip_code = "—"

            f1 = _normalize(row.get("fighter1", ""))
            f2 = _normalize(row.get("fighter2", ""))
            wc = fight_lookup.get((f1, f2))
            if not wc:
                wc = fighter_wc.get(f1) or fighter_wc.get(f2) or "Unknown"

            fights.append({
                "date": row.get("date", ""),
                "fighter1": row.get("fighter1", ""),
                "fighter2": row.get("fighter2", ""),
                "pick": row.get("pick", ""),
                "pick_odds": round(pick_odds),
                "pick_prob": round(pick_prob * 100, 1),
                "correct": correct,
                "bet": actually_bet,
                "skip_code": skip_code,
                "pnl": pnl,
                "weight_class": wc,
            })

    return fights


def _analyze_year(results_path: Path, bets_path: Path, config: dict):
    bets_filter = parse_bets_txt(str(bets_path))
    rows = parse_csv(str(results_path), bets=bets_filter)
    if not rows:
        return None

    # Also parse with weight class info
    rows_with_wc = _parse_csv_with_weight_class(results_path, bets_filter)

    # Overall stats across ALL bets (not filtered by edge tier)
    n = len(rows)
    w = sum(1 for r in rows if r.correct)
    profit = sum(r.pnl for r in rows)
    overall = {
        "n": n,
        "w": w,
        "l": n - w,
        "win_rate": round(w / n * 100, 1) if n else 0,
        "profit": round(profit, 2),
        "roi": round(profit / n * 100, 1) if n else 0,
    }

    return {
        "n_bets": len(rows),
        "overall": overall,
        "odds_buckets": _build_odds_buckets(rows),
        "edge_tiers": _build_edge_tiers(rows),
        "weighted_roi": _build_weighted_roi(rows, config),
        "weight_class": _build_weight_class_performance(rows_with_wc),
        "fights": _build_fights_table(results_path, bets_filter),
    }


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.get("/bucket-analysis")
async def bucket_analysis():
    config = json.loads(_CONFIG_PATH.read_text())
    result = {"config": config}
    for year, paths in DATASETS.items():
        data = _analyze_year(paths["results"], paths["bets"], config)
        if data:
            result[year] = data
    return result
