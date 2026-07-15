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
POSITIVE_GOLDEN_BUCKETS = {
    "golden_elo_not_expensive",
    "golden_elo_plus_trait_support",
    "golden_elo_plus_cardio",
}


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


@lru_cache(maxsize=4)
def _sidecar_fighter_index(sidecar_path: str, mtime_ns: int) -> dict[str, dict[str, Any]]:
    """Map canonical name -> identity/rating fields from the sidecar fighters table."""
    path = Path(sidecar_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT fighter_id, full_name, elo_current, elo_peak "
            "FROM fighters WHERE full_name IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    index: dict[str, dict[str, Any]] = {}
    aliases = _aliases()
    for row in rows:
        normalized = canonical_name(row["full_name"], aliases)
        # Prefer the row with the highest current ELO on a name collision
        # (e.g. namesake / whitespace-variant duplicates in the source data).
        existing = index.get(normalized)
        cur = row["elo_current"]
        if existing is not None and (cur is None or (existing["elo_current"] or -1) >= (cur or -1)):
            continue
        index[normalized] = {
            "fighter_id": row["fighter_id"],
            "full_name": row["full_name"],
            "elo_current": cur,
            "elo_peak": row["elo_peak"],
        }
    return index


def _fighter_record(name: str, *, sidecar_path: Path = SIDECAR_PATH) -> dict[str, Any] | None:
    if not sidecar_path.exists():
        return None
    index = _sidecar_fighter_index(str(sidecar_path), sidecar_path.stat().st_mtime_ns)
    return index.get(canonical_name(name, _aliases()))


def current_elo(name: str, *, sidecar_path: Path = SIDECAR_PATH) -> int | None:
    """Public: current cross-promotion ELO for a fighter, or None if unknown."""
    rec = _fighter_record(name, sidecar_path=sidecar_path)
    if not rec or rec.get("elo_current") is None:
        return None
    return int(round(rec["elo_current"]))


def peak_elo(name: str, *, sidecar_path: Path = SIDECAR_PATH) -> int | None:
    """Public: peak ELO for a fighter, or None if unknown."""
    rec = _fighter_record(name, sidecar_path=sidecar_path)
    if not rec or rec.get("elo_peak") is None:
        return None
    return int(round(rec["elo_peak"]))


def elo_history(name: str, *, sidecar_path: Path = SIDECAR_PATH) -> list[dict[str, Any]]:
    """
    Chronological pre-fight ELO snapshots for a fighter from the sidecar fights
    table, with a final point at the fighter's current ELO. Each entry:
    {date, elo, opponent, result, division, promotion, is_ufc}.
    """
    rec = _fighter_record(name, sidecar_path=sidecar_path)
    if not rec:
        return []
    fighter_id = rec["fighter_id"]
    conn = sqlite3.connect(f"file:{sidecar_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT fight_date, event_date, division, promotion,
                   fighter_red_id, fighter_blue_id,
                   fighter_red_name, fighter_blue_name,
                   fighter_red_elo, fighter_blue_elo, winner_id
            FROM fights
            WHERE fighter_red_id = ? OR fighter_blue_id = ?
            """,
            (fighter_id, fighter_id),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    history: list[dict[str, Any]] = []
    for row in rows:
        is_red = row["fighter_red_id"] == fighter_id
        elo = row["fighter_red_elo"] if is_red else row["fighter_blue_elo"]
        opp_elo = row["fighter_blue_elo"] if is_red else row["fighter_red_elo"]
        if elo is None:
            continue
        date_str = row["fight_date"] or row["event_date"]
        if not date_str:
            continue
        opponent = row["fighter_blue_name"] if is_red else row["fighter_red_name"]
        winner_id = row["winner_id"]
        if winner_id is None:
            result = None
        elif winner_id == fighter_id:
            result = "W"
        else:
            result = "L"
        promotion = row["promotion"] or ""
        history.append({
            "date": date_str,
            "elo": int(round(elo)),
            "opp_elo": int(round(opp_elo)) if opp_elo is not None else None,
            "opponent": opponent,
            "result": result,
            "division": row["division"],
            "promotion": promotion,
            "is_ufc": "ultimate fighting" in promotion.lower() or promotion.strip().upper() == "UFC",
        })

    history.sort(key=lambda h: h["date"])

    # Append a final point at the current ELO so the line ends at today's rating.
    if rec.get("elo_current") is not None:
        last_date = history[-1]["date"] if history else None
        history.append({
            "date": "current",
            "elo": int(round(rec["elo_current"])),
            "opponent": None,
            "result": None,
            "division": None,
            "promotion": None,
            "is_ufc": None,
            "after_date": last_date,
        })
    return history


@lru_cache(maxsize=32)
def _historical_rows(context_pool_path: str, mtime_ns: int, golden_only: bool) -> list[dict[str, Any]]:
    path = Path(context_pool_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = """
            WHERE pick_correct IS NOT NULL
        """
    if golden_only:
        where += """
              AND bet = 0
              AND pick_prob >= 0.50
              AND pick_prob < 0.65
              AND pick_odds > -300
        """
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
            """
            + where
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


def _has_offset_trait_support(deltas: dict[str, Any], cfg: dict[str, Any]) -> bool:
    threshold = float(cfg.get("trait_support_min_diff", 10))
    return (
        (deltas.get("cardio_score_diff") or 0) >= threshold
        or (deltas.get("striking_efficiency_score_diff") or 0) >= threshold
    )


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


def _row_number(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _build_review_context(
    *,
    pick_elo_diff: float | None,
    review_bucket: str | None,
    review_tier: int | str | None,
    review_base: str | None,
    cfg: dict[str, Any],
    context_pool_path: Path,
    trait_support: bool,
    cardio_support: bool,
    offset_trait_support: bool,
) -> dict[str, Any]:
    review_stats = None
    review_label = None
    if review_bucket:
        review_stats = _review_historical_stats(review_bucket, cfg, context_pool_path=context_pool_path)
        review_label = _format_label(review_base or "Review", review_stats)
    return {
        "pick_elo_diff": pick_elo_diff,
        "review_bucket": review_bucket,
        "review_tier": review_tier,
        "review_label": review_label,
        "review_stats": review_stats,
        "trait_support": trait_support,
        "cardio_support": cardio_support,
        "offset_trait_support": offset_trait_support,
    }


def _golden_historical_stats(tier: int | str, cfg: dict[str, Any], *, context_pool_path: Path = CONTEXT_POOL_PATH) -> dict[str, Any] | None:
    if not context_pool_path.exists():
        return None
    rows = _historical_rows(str(context_pool_path), context_pool_path.stat().st_mtime_ns, True)
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


def _review_historical_stats(review_bucket: str, cfg: dict[str, Any], *, context_pool_path: Path = CONTEXT_POOL_PATH) -> dict[str, Any] | None:
    if review_bucket.startswith("golden_elo_"):
        tier: int | str = (
            3
            if review_bucket == "golden_elo_plus_cardio"
            else 2
            if review_bucket == "golden_elo_plus_trait_support"
            else 1
        )
        return _golden_historical_stats(tier, cfg, context_pool_path=context_pool_path)

    if not context_pool_path.exists():
        return None
    rows = _historical_rows(str(context_pool_path), context_pool_path.stat().st_mtime_ns, False)
    if not rows:
        return None

    if review_bucket == "neutral_elo_favorite":
        matched = [
            row for row in rows
            if _row_number(row, "pick_odds", 0) < 0
            and -50 < _row_number(row, "pick_elo_diff", 9999) < 50
        ]
    elif review_bucket == "neutral_elo_underdog":
        matched = [
            row for row in rows
            if _row_number(row, "pick_odds", 0) > 0
            and -50 < _row_number(row, "pick_elo_diff", 9999) < 50
        ]
    elif review_bucket == "favorite_negative_elo_midprice_no_offset":
        matched = [
            row for row in rows
            if -300 < (row.get("pick_odds") or 0) < 0
            and (row.get("pick_elo_diff") or 9999) <= -50
            and not (
                _trait_confident(row, cfg)
                and _has_offset_trait_support(row.get("deltas") or {}, cfg)
            )
        ]
    elif review_bucket == "favorite_negative_elo_no_offset":
        matched = [
            row for row in rows
            if (row.get("pick_odds") or 0) < 0
            and (row.get("pick_elo_diff") or 9999) <= -50
            and not (
                _trait_confident(row, cfg)
                and _has_offset_trait_support(row.get("deltas") or {}, cfg)
            )
        ]
    elif review_bucket == "elo_against_100":
        matched = [row for row in rows if (row.get("pick_elo_diff") or 9999) <= -100]
    elif review_bucket == "elo_against_tier_1a":
        matched = [
            row for row in rows
            if -100 < (row.get("pick_elo_diff") or 9999) <= -50
            and _trait_confident(row, cfg)
            and _has_offset_trait_support(row.get("deltas") or {}, cfg)
        ]
    elif review_bucket == "trait_offset_elo_against":
        matched = [
            row for row in rows
            if (row.get("pick_elo_diff") or 9999) <= -50
            and _trait_confident(row, cfg)
            and _has_offset_trait_support(row.get("deltas") or {}, cfg)
        ]
    else:
        matched = [row for row in rows if (row.get("pick_elo_diff") or 9999) <= -50]
    return _cohort_stats(matched)


def _format_label(base: str, stats: dict[str, Any] | None) -> str:
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


def evaluate_elo_review_context(
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
    pick_name = fighter1_name if pick_slot == "fighter1" else fighter2_name
    opp_name = fighter2_name if pick_slot == "fighter1" else fighter1_name
    pick_elo = _current_elo(pick_name, sidecar_path=sidecar_path)
    opp_elo = _current_elo(opp_name, sidecar_path=sidecar_path)
    if pick_elo is None or opp_elo is None:
        return _build_review_context(
            pick_elo_diff=None,
            review_bucket=None,
            review_tier=None,
            review_base=None,
            cfg=cfg,
            context_pool_path=context_pool_path,
            trait_support=False,
            cardio_support=False,
            offset_trait_support=False,
        )

    pick_elo_diff = round(pick_elo - opp_elo, 1)
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
    has_offset_trait_support = bool(
        traits
        and _trait_confident(traits, cfg)
        and _has_offset_trait_support(traits, cfg)
    )
    is_favorite = pick_odds is not None and pick_odds < 0
    is_midpriced_favorite = pick_odds is not None and -300 < pick_odds < 0

    review_bucket = None
    review_tier = None
    review_base = None
    if pick_elo_diff <= -50:
        review_bucket = "elo_against_50"
        review_tier = 1
        review_base = "ELO Against Tier 1"
        if pick_elo_diff <= -100:
            review_bucket = "elo_against_100"
            review_tier = 2
            review_base = "ELO Against Tier 2"
        if has_offset_trait_support and pick_elo_diff > -100:
            review_bucket = "elo_against_tier_1a"
            review_tier = "-1A"
            review_base = "ELO Against Tier -1A"
        elif has_offset_trait_support:
            review_bucket = "trait_offset_elo_against"
            review_tier = 3
            review_base = "Trait Offset Tier 3"
        elif is_midpriced_favorite:
            review_bucket = "favorite_negative_elo_midprice_no_offset"
            review_tier = "F-"
            review_base = "Favorite ELO Fade"
        elif is_favorite:
            review_bucket = "favorite_negative_elo_no_offset"
            review_tier = "F"
            review_base = "Favorite Negative ELO Caution"
    elif -50 < pick_elo_diff < 50:
        if is_favorite:
            review_bucket = "neutral_elo_favorite"
            review_tier = "F0"
            review_base = "Neutral ELO Favorite"
        elif pick_odds is not None and pick_odds > 0:
            review_bucket = "neutral_elo_underdog"
            review_tier = "D0"
            review_base = "Neutral ELO Underdog"

    if not cfg.get("enabled", True):
        return _build_review_context(
            pick_elo_diff=pick_elo_diff,
            review_bucket=review_bucket,
            review_tier=review_tier,
            review_base=review_base,
            cfg=cfg,
            context_pool_path=context_pool_path,
            trait_support=has_trait_support,
            cardio_support=has_cardio_support,
            offset_trait_support=has_offset_trait_support,
        )
    if not (cfg["confidence_min"] <= pick_model_prob < cfg["confidence_max"]):
        return _build_review_context(
            pick_elo_diff=pick_elo_diff,
            review_bucket=review_bucket,
            review_tier=review_tier,
            review_base=review_base,
            cfg=cfg,
            context_pool_path=context_pool_path,
            trait_support=has_trait_support,
            cardio_support=has_cardio_support,
            offset_trait_support=has_offset_trait_support,
        )
    if pick_odds is None or pick_odds <= cfg["min_pick_odds"]:
        return _build_review_context(
            pick_elo_diff=pick_elo_diff,
            review_bucket=review_bucket,
            review_tier=review_tier,
            review_base=review_base,
            cfg=cfg,
            context_pool_path=context_pool_path,
            trait_support=has_trait_support,
            cardio_support=has_cardio_support,
            offset_trait_support=has_offset_trait_support,
        )
    if pick_elo_diff < cfg["min_pick_elo_diff"]:
        return _build_review_context(
            pick_elo_diff=pick_elo_diff,
            review_bucket=review_bucket,
            review_tier=review_tier,
            review_base=review_base,
            cfg=cfg,
            context_pool_path=context_pool_path,
            trait_support=has_trait_support,
            cardio_support=has_cardio_support,
            offset_trait_support=has_offset_trait_support,
        )

    tier = 1
    review_bucket = "golden_elo_not_expensive"
    review_base = "Golden ELO Tier 1"
    if pick_elo_diff >= cfg["tier_2_min_elo_diff"] and has_trait_support:
        tier = 2
        review_bucket = "golden_elo_plus_trait_support"
        review_base = "Golden ELO Tier 2"
        if has_cardio_support:
            tier = 3
            review_bucket = "golden_elo_plus_cardio"
            review_base = "Golden ELO Tier 3"

    return _build_review_context(
        pick_elo_diff=pick_elo_diff,
        review_bucket=review_bucket,
        review_tier=tier,
        review_base=review_base,
        cfg=cfg,
        context_pool_path=context_pool_path,
        trait_support=has_trait_support,
        cardio_support=has_cardio_support,
        offset_trait_support=has_offset_trait_support,
    )


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
    review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if review_context is None:
        review_context = evaluate_elo_review_context(
            fighter1_name=fighter1_name,
            fighter2_name=fighter2_name,
            pick_slot=pick_slot,
            pick_model_prob=pick_model_prob,
            pick_odds=pick_odds,
            as_of_date=as_of_date,
            sidecar_path=sidecar_path,
            context_pool_path=context_pool_path,
            trait_snapshot_path=trait_snapshot_path,
            main_db_path=main_db_path,
        )
    return {
        **review_context,
        "reopen": review_context.get("review_bucket") in POSITIVE_GOLDEN_BUCKETS,
    }
