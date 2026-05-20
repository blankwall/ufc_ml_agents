from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from backtest.elo_analysis import DEFAULT_ALIAS_SOURCES, canonical_name, load_aliases

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "betting_config.json"
SIDECAR_PATH = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
CONTEXT_POOL_PATH = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"
TRAIT_SNAPSHOT_PATH = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"
MAIN_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"


def load_golden_elo_config() -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "confidence_min": 0.50,
        "confidence_max": 0.65,
        "min_pick_odds": -300,
        "min_pick_elo_diff": 50,
        "tier_2_min_elo_diff": 100,
        "min_trait_confidence": 0.60,
        "trait_support_min_diff": 10,
        "tier_3_cardio_min_diff": 10,
    }
    if not CONFIG_PATH.exists():
        return defaults
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return defaults
    return {**defaults, **(cfg.get("golden_elo_reopen") or {})}


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    return load_aliases(DEFAULT_ALIAS_SOURCES)


@lru_cache(maxsize=4)
def _sidecar_fighters(sidecar_path: str, mtime_ns: int) -> dict[str, float]:
    path = Path(sidecar_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT full_name, elo_current FROM fighters WHERE full_name IS NOT NULL AND elo_current IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    fighters: dict[str, float] = {}
    aliases = _aliases()
    for row in rows:
        normalized = canonical_name(row["full_name"], aliases)
        try:
            fighters[normalized] = float(row["elo_current"])
        except (TypeError, ValueError):
            continue
    return fighters


def _current_elo(name: str, *, sidecar_path: Path = SIDECAR_PATH) -> float | None:
    if not sidecar_path.exists():
        return None
    fighters = _sidecar_fighters(str(sidecar_path), sidecar_path.stat().st_mtime_ns)
    return fighters.get(canonical_name(name, _aliases()))


@lru_cache(maxsize=32)
def _historical_rows(context_pool_path: str, mtime_ns: int) -> list[dict[str, Any]]:
    path = Path(context_pool_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                p.pick_correct,
                p.actual_pnl,
                p.pick_prob,
                p.pick_elo_diff,
                p.pick_odds,
                e.data_json
            FROM backtest_fight_pool p
            LEFT JOIN evidence_items e
              ON e.fight_pool_id = p.id
             AND e.evidence_type = 'trait_delta'
            WHERE pick_correct IS NOT NULL
              AND bet = 0
              AND pick_prob >= 0.50
              AND pick_prob < 0.65
              AND pick_odds > -300
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = {}
        if row["data_json"]:
            try:
                payload = json.loads(row["data_json"])
            except json.JSONDecodeError:
                payload = {}
        parsed_rows.append(
            {
                "pick_correct": row["pick_correct"],
                "actual_pnl": row["actual_pnl"],
                "pick_prob": row["pick_prob"],
                "pick_elo_diff": row["pick_elo_diff"],
                "pick_odds": row["pick_odds"],
                "trait_confidence": payload.get("trait_confidence"),
                "opponent_trait_confidence": payload.get("opponent_trait_confidence"),
                "deltas": payload.get("deltas") or {},
            }
        )
    return parsed_rows


def _has_primary_trait_support(deltas: dict[str, Any], cfg: dict[str, Any]) -> bool:
    threshold = float(cfg.get("trait_support_min_diff", 10))
    return (
        (deltas.get("cardio_score_diff") or 0) >= threshold
        or (deltas.get("striking_efficiency_score_diff") or 0) >= threshold
        or (deltas.get("defensive_exposure_score_diff") or 0) <= -threshold
    )


def _has_cardio_support(deltas: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (deltas.get("cardio_score_diff") or 0) >= float(cfg.get("tier_3_cardio_min_diff", 10))


def _trait_confident(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    minimum = float(cfg.get("min_trait_confidence", 0.60))
    return (row.get("trait_confidence") or 0) >= minimum and (row.get("opponent_trait_confidence") or 0) >= minimum


def _cohort_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    n = len(rows)
    if not n:
        return None
    wins = sum(1 for row in rows if row["pick_correct"] == 1)
    losses = sum(1 for row in rows if row["pick_correct"] == 0)
    profit = sum(float(row["actual_pnl"] or 0.0) for row in rows)
    return {
        "n": n,
        "wins": int(wins),
        "losses": int(losses),
        "profit": round(profit, 3),
        "roi_pct": round((profit / n) * 100, 1) if n else None,
    }


def _historical_stats(tier: int, cfg: dict[str, Any], *, context_pool_path: Path = CONTEXT_POOL_PATH) -> dict[str, Any] | None:
    if not context_pool_path.exists():
        return None
    rows = _historical_rows(str(context_pool_path), context_pool_path.stat().st_mtime_ns)
    if not rows:
        return None

    if tier == 3:
        matched = [
            row for row in rows
            if (row.get("pick_elo_diff") or -9999) >= float(cfg["tier_2_min_elo_diff"])
            and _trait_confident(row, cfg)
            and _has_primary_trait_support(row.get("deltas") or {}, cfg)
            and _has_cardio_support(row.get("deltas") or {}, cfg)
        ]
    elif tier == 2:
        matched = [
            row for row in rows
            if (row.get("pick_elo_diff") or -9999) >= float(cfg["tier_2_min_elo_diff"])
            and _trait_confident(row, cfg)
            and _has_primary_trait_support(row.get("deltas") or {}, cfg)
        ]
    else:
        matched = [
            row for row in rows
            if (row.get("pick_elo_diff") or -9999) >= float(cfg["min_pick_elo_diff"])
        ]
    return _cohort_stats(matched)


def _format_label(tier: int, stats: dict[str, Any] | None) -> str:
    base = f"Golden ELO Tier {tier}"
    if not stats or not stats.get("n"):
        return base
    roi = stats.get("roi_pct")
    roi_text = "n/a" if roi is None else f"{roi:+.1f}% ROI"
    return f"{base} · Historical {stats['wins']}-{stats['losses']} · {roi_text}"


def _normalize_as_of_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return None


@lru_cache(maxsize=4)
def _fighter_ids(main_db_path: str, mtime_ns: int) -> dict[str, int]:
    path = Path(main_db_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id, name FROM fighters WHERE name IS NOT NULL").fetchall()
    finally:
        conn.close()
    return {str(row["name"]): int(row["id"]) for row in rows}


def _fighter_id(name: str, *, main_db_path: Path = MAIN_DB_PATH) -> int | None:
    if not main_db_path.exists():
        return None
    return _fighter_ids(str(main_db_path), main_db_path.stat().st_mtime_ns).get(name)


def _trait_pair_delta(
    *,
    fighter1_name: str,
    fighter2_name: str,
    pick_slot: str,
    as_of_date: date | datetime | str | None,
    trait_snapshot_path: Path = TRAIT_SNAPSHOT_PATH,
    main_db_path: Path = MAIN_DB_PATH,
) -> dict[str, Any] | None:
    if not trait_snapshot_path.exists():
        return None
    pick_name = fighter1_name if pick_slot == "fighter1" else fighter2_name
    opp_name = fighter2_name if pick_slot == "fighter1" else fighter1_name
    pick_id = _fighter_id(pick_name, main_db_path=main_db_path)
    opp_id = _fighter_id(opp_name, main_db_path=main_db_path)
    if pick_id is None or opp_id is None:
        return None

    as_of_iso = _normalize_as_of_date(as_of_date)
    conn = sqlite3.connect(f"file:{trait_snapshot_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if as_of_iso is None:
            row = conn.execute(
                """
                SELECT *
                FROM v_trait_pair_deltas
                WHERE fighter_id = ?
                  AND opponent_id = ?
                ORDER BY as_of_date DESC, main_fight_id DESC
                LIMIT 1
                """,
                (pick_id, opp_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT *
                FROM v_trait_pair_deltas
                WHERE fighter_id = ?
                  AND opponent_id = ?
                  AND as_of_date <= ?
                ORDER BY as_of_date DESC, main_fight_id DESC
                LIMIT 1
                """,
                (pick_id, opp_id, as_of_iso),
            ).fetchone()
    finally:
        conn.close()

    return dict(row) if row is not None else None


def evaluate_golden_elo_reopen(
    *,
    fighter1_name: str,
    fighter2_name: str,
    pick_slot: str,
    pick_model_prob: float,
    pick_odds: int | None,
    as_of_date: date | datetime | str | None = None,
    sidecar_path: Path = SIDECAR_PATH,
    context_pool_path: Path = CONTEXT_POOL_PATH,
    trait_snapshot_path: Path = TRAIT_SNAPSHOT_PATH,
    main_db_path: Path = MAIN_DB_PATH,
) -> dict[str, Any]:
    cfg = load_golden_elo_config()
    if not cfg.get("enabled", True):
        return {"reopen": False, "pick_elo_diff": None}

    if not (cfg["confidence_min"] <= pick_model_prob < cfg["confidence_max"]):
        return {"reopen": False, "pick_elo_diff": None}
    if pick_odds is None or pick_odds <= cfg["min_pick_odds"]:
        return {"reopen": False, "pick_elo_diff": None}

    pick_name = fighter1_name if pick_slot == "fighter1" else fighter2_name
    opp_name = fighter2_name if pick_slot == "fighter1" else fighter1_name
    pick_elo = _current_elo(pick_name, sidecar_path=sidecar_path)
    opp_elo = _current_elo(opp_name, sidecar_path=sidecar_path)
    if pick_elo is None or opp_elo is None:
        return {"reopen": False, "pick_elo_diff": None}

    pick_elo_diff = round(pick_elo - opp_elo, 1)
    if pick_elo_diff < cfg["min_pick_elo_diff"]:
        return {"reopen": False, "pick_elo_diff": pick_elo_diff}

    traits = _trait_pair_delta(
        fighter1_name=fighter1_name,
        fighter2_name=fighter2_name,
        pick_slot=pick_slot,
        as_of_date=as_of_date,
        trait_snapshot_path=trait_snapshot_path,
        main_db_path=main_db_path,
    )
    has_trait_support = bool(
        traits
        and _trait_confident(traits, cfg)
        and _has_primary_trait_support(traits, cfg)
    )
    has_cardio_support = bool(
        has_trait_support
        and traits
        and _has_cardio_support(traits, cfg)
    )

    tier = 1
    review_bucket = "golden_elo_not_expensive"
    if pick_elo_diff >= cfg["tier_2_min_elo_diff"] and has_trait_support:
        tier = 2
        review_bucket = "golden_elo_plus_trait_support"
        if has_cardio_support:
            tier = 3
            review_bucket = "golden_elo_plus_cardio"

    stats = _historical_stats(tier, cfg, context_pool_path=context_pool_path)

    return {
        "reopen": True,
        "pick_elo_diff": pick_elo_diff,
        "review_bucket": review_bucket,
        "review_tier": tier,
        "review_label": _format_label(tier, stats),
        "review_stats": stats,
        "trait_support": has_trait_support,
        "cardio_support": has_cardio_support,
    }
