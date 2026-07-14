from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FASTAPI_DIR = Path(__file__).resolve().parent.parent
for import_path in (ROOT_DIR, FASTAPI_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from database.schema import Fighter  # noqa: E402
from features.matchup_features import MatchupFeatureExtractor  # noqa: E402
from services.fighter_snapshot import build_fighter_snapshot  # noqa: E402
from services.predict_service import (  # noqa: E402
    FIGHTER_ALIASES,
    _fight_count_as_of,
    _prediction_cutoff_datetime,
    _resolve_fighter,
    _score_row,
)

DEFAULT_CONTEXT_POOL = ROOT_DIR / "data" / "enrichment" / "context_pool.sqlite"
_DB_PATH = ROOT_DIR / "data" / "ufc_database.db"
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)

TRAIT_DELTA_FIELDS = (
    "cardio_score_diff",
    "striking_efficiency_score_diff",
    "defensive_exposure_score_diff",
    "offensive_control_score_diff",
    "anti_control_score_diff",
    "scramble_score_diff",
    "striking_pressure_score_diff",
    "finishing_threat_score_diff",
    "grappling_threat_score_diff",
    "durability_risk_score_diff",
    "variance_score_diff",
    "experience_score_diff",
    "recent_form_score_diff",
)

_ALLOWED_COUNT_TABLES = {"backtest_fight_pool", "pattern_stats", "evidence_items"}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _as_float(value)
    return None if numeric is None else round(numeric, digits)


def _american_to_prob(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _market_prob_f1(fighter1_odds: int | None, fighter2_odds: int | None) -> float:
    if fighter1_odds is not None and fighter2_odds is not None:
        raw1 = _american_to_prob(fighter1_odds)
        raw2 = _american_to_prob(fighter2_odds)
        vig = raw1 + raw2
        return raw1 / vig if vig else 0.5
    if fighter1_odds is not None:
        return _american_to_prob(fighter1_odds)
    if fighter2_odds is not None:
        return 1 - _american_to_prob(fighter2_odds)
    return 0.5


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    if table_name not in _ALLOWED_COUNT_TABLES or not _table_exists(conn, table_name):
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _context_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _open_context_pool(path: Path) -> tuple[sqlite3.Connection | None, dict[str, Any]]:
    quality = {
        "context_pool_available": False,
        "context_pool_path": _context_source_path(path),
        "row_counts": {},
        "missing_fields": [],
        "reasons": [],
    }
    if not path.exists():
        quality["reasons"].append("context_pool.sqlite not found")
        return None, quality

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        quality["reasons"].append(f"context_pool open failed: {exc}")
        return None, quality

    quality["context_pool_available"] = True
    for table in sorted(_ALLOWED_COUNT_TABLES):
        quality["row_counts"][table] = _table_count(conn, table)
    return conn, quality


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("pick_correct", "bet", "female", "model_agrees_with_elo", "elo_pick_correct"):
        if key in data:
            data[key] = _bool_or_none(data.get(key))
    return data


def build_prediction_frame(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    session=None,
) -> dict[str, Any]:
    """Run the existing symmetric model path and orient fields around the model pick."""
    created_session = session is None
    session = session or _Session()
    try:
        f1_name = FIGHTER_ALIASES.get(fighter1, fighter1)
        f2_name = FIGHTER_ALIASES.get(fighter2, fighter2)
        f1: Optional[Fighter] = _resolve_fighter(session, f1_name)
        f2: Optional[Fighter] = _resolve_fighter(session, f2_name)
        missing = [name for name, fighter in ((fighter1, f1), (fighter2, f2)) if fighter is None]
        if missing:
            raise LookupError(f"Fighter(s) not found: {', '.join(missing)}")

        market_prob_f1 = _market_prob_f1(fighter1_odds, fighter2_odds)
        as_of = _prediction_cutoff_datetime(fight_date)
        pred = _score_row(
            session,
            MatchupFeatureExtractor(session),
            f1.id,
            f2.id,
            market_prob_f1,
            as_of_date=as_of,
        )
        model_prob_f1 = float(pred["model_prob_f1"])

        if model_prob_f1 >= 0.5:
            pick_slot = "fighter1"
            opponent_slot = "fighter2"
            model_pick = fighter1
            model_pick_db_name = f1.name
            opponent_db_name = f2.name
            pick_prob = model_prob_f1
            market_prob = market_prob_f1
            pick_odds = fighter1_odds
        else:
            pick_slot = "fighter2"
            opponent_slot = "fighter1"
            model_pick = fighter2
            model_pick_db_name = f2.name
            opponent_db_name = f1.name
            pick_prob = 1 - model_prob_f1
            market_prob = 1 - market_prob_f1
            pick_odds = fighter2_odds

        f1_count = _fight_count_as_of(session, f1.id, as_of)
        f2_count = _fight_count_as_of(session, f2.id, as_of)
        edge = pick_prob - market_prob

        return {
            "request": {
                "fighter1": fighter1,
                "fighter2": fighter2,
                "fight_date": _iso_date(fight_date),
                "fighter1_odds": fighter1_odds,
                "fighter2_odds": fighter2_odds,
            },
            "as_of": as_of,
            "resolved": {
                "fighter1_db_name": f1.name,
                "fighter2_db_name": f2.name,
                "model_pick_db_name": model_pick_db_name,
                "opponent_db_name": opponent_db_name,
            },
            "fight_counts": {"fighter1": f1_count, "fighter2": f2_count},
            "model_context": {
                "model_pick": model_pick,
                "model_pick_db_name": model_pick_db_name,
                "pick_slot": pick_slot,
                "opponent_slot": opponent_slot,
                "pick_prob": round(pick_prob, 4),
                "pick_odds": pick_odds,
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "model_prob_f1": round(model_prob_f1, 4),
                "model_prob_f2": round(1 - model_prob_f1, 4),
                "model_source": pred.get("model_source"),
            },
        }
    finally:
        if created_session:
            session.close()


def _safe_snapshot(fighter_name: str, *, as_of: datetime | None, session, quality: dict[str, Any]) -> dict[str, Any]:
    try:
        return build_fighter_snapshot(fighter_name, as_of=as_of, session=session)
    except Exception as exc:  # surface the reason in data_quality instead of failing the endpoint
        quality.setdefault("snapshot_errors", []).append({"fighter": fighter_name, "error": str(exc)})
        return {"query_name": fighter_name, "resolved": False, "note": str(exc)}


def _trait_value(snapshot: dict[str, Any], field: str) -> float | None:
    return _as_float((snapshot.get("qualitative") or {}).get(field))


def _extract_current_context(frame: dict[str, Any], *, session, quality: dict[str, Any]) -> dict[str, Any]:
    as_of = frame.get("as_of")
    resolved = frame["resolved"]
    f1_snapshot = _safe_snapshot(resolved["fighter1_db_name"], as_of=as_of, session=session, quality=quality)
    f2_snapshot = _safe_snapshot(resolved["fighter2_db_name"], as_of=as_of, session=session, quality=quality)
    pick_snapshot = f1_snapshot if frame["model_context"]["pick_slot"] == "fighter1" else f2_snapshot
    opp_snapshot = f2_snapshot if frame["model_context"]["pick_slot"] == "fighter1" else f1_snapshot

    pick_elo = (pick_snapshot.get("elo") or {}).get("elo_current")
    opp_elo = (opp_snapshot.get("elo") or {}).get("elo_current")
    pick_elo_diff = None
    if pick_elo is not None and opp_elo is not None:
        pick_elo_diff = round(float(pick_elo) - float(opp_elo), 1)

    deltas: dict[str, float | None] = {}
    for diff_field in TRAIT_DELTA_FIELDS:
        base_field = diff_field.removesuffix("_diff")
        pick_value = _trait_value(pick_snapshot, base_field)
        opp_value = _trait_value(opp_snapshot, base_field)
        deltas[diff_field] = round(pick_value - opp_value, 1) if pick_value is not None and opp_value is not None else None

    pick_quality = pick_snapshot.get("qualitative") or {}
    opp_quality = opp_snapshot.get("qualitative") or {}
    current = {
        "pick_elo": _round(pick_elo, 1),
        "opponent_elo": _round(opp_elo, 1),
        "pick_elo_diff": pick_elo_diff,
        "model_agrees_with_elo": None if pick_elo_diff is None else pick_elo_diff >= 0,
        "cardio_score_diff": deltas.get("cardio_score_diff"),
        "trait_deltas": deltas,
        "trait_confidence": _round(pick_quality.get("trait_confidence"), 3),
        "opponent_trait_confidence": _round(opp_quality.get("trait_confidence"), 3),
        "snapshots_available": {
            "fighter1": bool(f1_snapshot.get("resolved")),
            "fighter2": bool(f2_snapshot.get("resolved")),
            "pick_traits": bool(pick_quality.get("available")),
            "opponent_traits": bool(opp_quality.get("available")),
        },
    }

    for key, value in {
        "pick_elo_diff": current["pick_elo_diff"],
        "trait_deltas": None if all(value is None for value in deltas.values()) else "present",
    }.items():
        if value is None:
            quality.setdefault("missing_fields", []).append(key)
    return current


def _trait_support(deltas: dict[str, float | None]) -> tuple[list[str], list[str]]:
    support: list[str] = []
    cautions: list[str] = []

    if (deltas.get("cardio_score_diff") or 0) >= 10:
        support.append("cardio_10_plus")
    elif deltas.get("cardio_score_diff") is not None and deltas["cardio_score_diff"] <= -10:
        cautions.append("cardio_minus_10")

    if (deltas.get("striking_efficiency_score_diff") or 0) >= 10:
        support.append("striking_efficiency_10_plus")
    elif deltas.get("striking_efficiency_score_diff") is not None and deltas["striking_efficiency_score_diff"] <= -10:
        cautions.append("striking_efficiency_minus_10")

    if deltas.get("defensive_exposure_score_diff") is not None:
        if deltas["defensive_exposure_score_diff"] <= -10:
            support.append("safer_defensive_exposure_10_plus")
        elif deltas["defensive_exposure_score_diff"] >= 10:
            cautions.append("defensive_exposure_plus_10")

    for field, label in (
        ("scramble_score_diff", "scramble_10_plus"),
        ("anti_control_score_diff", "anti_control_10_plus"),
        ("offensive_control_score_diff", "offensive_control_10_plus"),
        ("finishing_threat_score_diff", "finishing_threat_10_plus"),
    ):
        if (deltas.get(field) or 0) >= 10:
            support.append(label)

    primary = {
        "cardio_10_plus",
        "striking_efficiency_10_plus",
        "safer_defensive_exposure_10_plus",
    }
    if primary.intersection(support):
        support.insert(0, "primary_trait_support_any")
    elif (deltas.get("striking_pressure_score_diff") or 0) >= 10:
        cautions.append("pressure_only_weak_support")

    return list(dict.fromkeys(support)), list(dict.fromkeys(cautions))


def _pattern_hits(frame: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    model = frame["model_context"]
    pick_prob = model["pick_prob"]
    pick_odds = model.get("pick_odds")
    pick_elo_diff = current.get("pick_elo_diff")
    labels: list[str] = []
    elo_support: list[str] = []

    if pick_elo_diff is None:
        labels.append("missing_elo")
    else:
        if pick_elo_diff >= 50:
            labels.append("elo_50_plus")
            elo_support.append("elo_50_plus")
        if pick_elo_diff >= 100:
            labels.append("elo_100_plus")
            elo_support.append("elo_100_plus")
        if pick_elo_diff > 0:
            labels.append("model_pick_higher_elo")
        elif pick_elo_diff < 0:
            labels.append("model_pick_lower_elo")
        if pick_odds is not None and pick_odds > 0 and pick_elo_diff < 0:
            labels.append("underdog_elo_against")
        elif pick_odds is not None and pick_odds > 0 and pick_elo_diff > 0:
            labels.append("underdog_elo_support")

        mid_confidence = 0.50 <= pick_prob < 0.65
        not_expensive = pick_odds is not None and pick_odds > -300
        if mid_confidence and pick_elo_diff >= 50:
            labels.append("skip_50_65_elo_50_plus")
        if mid_confidence and pick_elo_diff >= 100:
            labels.append("skip_50_65_elo_100_plus")
        if mid_confidence and pick_elo_diff >= 50 and not_expensive:
            labels.append("skip_50_65_elo_50_plus_not_expensive")
            labels.append("golden_elo_not_expensive")

    trait_support, trait_cautions = _trait_support(current.get("trait_deltas") or {})
    if "underdog_elo_against" in labels and "primary_trait_support_any" in trait_support:
        labels.append("trait_offset_elo_against")

    return {
        "labels": list(dict.fromkeys(labels)),
        "elo_support": elo_support,
        "trait_support": trait_support,
        "trait_cautions": trait_cautions,
    }


def _risk_flags(frame: dict[str, Any], current: dict[str, Any], hits: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    pick_elo_diff = current.get("pick_elo_diff")
    if pick_elo_diff is None:
        flags.append("missing_elo")
    elif pick_elo_diff <= -100:
        flags.append("elo_disagrees_strong")
    elif pick_elo_diff <= -50:
        flags.append("elo_disagrees_moderate")

    if frame["model_context"].get("pick_odds") is not None and frame["model_context"]["pick_odds"] <= -300:
        flags.append("expensive_favorite")

    counts = frame.get("fight_counts") or {}
    if min(counts.get("fighter1", 0), counts.get("fighter2", 0)) < 3:
        flags.append("thin_sample")

    trait_conf = current.get("trait_confidence")
    opp_trait_conf = current.get("opponent_trait_confidence")
    if trait_conf is None or opp_trait_conf is None:
        flags.append("missing_traits")
    elif trait_conf < 0.6 or opp_trait_conf < 0.6:
        flags.append("trait_low_confidence")

    flags.extend(hits.get("trait_cautions") or [])
    return list(dict.fromkeys(flags))


def _aggregate_sql(conn: sqlite3.Connection, where_sql: str) -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN pick_correct = 1 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pick_correct = 0 THEN 1 ELSE 0 END) AS losses,
            SUM(COALESCE(actual_pnl, 0.0)) AS profit,
            AVG(pick_prob) AS avg_confidence,
            AVG(edge) AS avg_edge,
            AVG(pick_elo_diff) AS avg_elo_diff
        FROM backtest_fight_pool
        WHERE pick_correct IS NOT NULL AND ({where_sql})
        """
    ).fetchone()
    n = int(row["n"] or 0)
    wins = int(row["wins"] or 0)
    profit = float(row["profit"] or 0.0)
    return {
        "n": n,
        "wins": wins,
        "losses": int(row["losses"] or 0),
        "win_rate": round(wins / n, 4) if n else None,
        "profit": round(profit, 4),
        "roi": round(profit / n, 4) if n else None,
        "avg_confidence": _round(row["avg_confidence"]),
        "avg_edge": _round(row["avg_edge"]),
        "avg_elo_diff": _round(row["avg_elo_diff"], 1),
    }


_COHORT_SQL = {
    "golden_elo_not_expensive": "bet = 0 AND pick_prob >= 0.50 AND pick_prob < 0.65 AND pick_elo_diff >= 50 AND pick_odds > -300",
    "elo_50_plus": "pick_elo_diff >= 50",
    "elo_100_plus": "pick_elo_diff >= 100",
    "skip_50_65_elo_50_plus": "bet = 0 AND pick_prob >= 0.50 AND pick_prob < 0.65 AND pick_elo_diff >= 50",
    "skip_50_65_elo_100_plus": "bet = 0 AND pick_prob >= 0.50 AND pick_prob < 0.65 AND pick_elo_diff >= 100",
    "skip_50_65_elo_50_plus_not_expensive": "bet = 0 AND pick_prob >= 0.50 AND pick_prob < 0.65 AND pick_elo_diff >= 50 AND pick_odds > -300",
    "model_pick_lower_elo": "pick_elo_diff < 0",
    "model_pick_higher_elo": "pick_elo_diff > 0",
    "underdog_elo_against": "pick_odds > 0 AND pick_elo_diff < 0",
    "underdog_elo_support": "pick_odds > 0 AND pick_elo_diff > 0",
}


def _historical_bucket_stats(conn: sqlite3.Connection | None, labels: Iterable[str]) -> dict[str, Any]:
    if conn is None or not _table_exists(conn, "backtest_fight_pool"):
        return {}

    ordered = list(dict.fromkeys(label for label in labels if label in _COHORT_SQL or label.startswith("skip_") or label in {"model_pick_lower_elo", "underdog_elo_against", "underdog_elo_support", "model_pick_higher_elo"}))
    stats: dict[str, Any] = {}

    if ordered and _table_exists(conn, "pattern_stats"):
        placeholders = ", ".join("?" for _ in ordered)
        rows = conn.execute(
            f"SELECT * FROM pattern_stats WHERE pattern_name IN ({placeholders})",
            ordered,
        ).fetchall()
        for row in rows:
            name = row["pattern_name"]
            stats[name] = {
                "name": name,
                "description": row["description"],
                "n": int(row["sample_size"] or 0),
                "wins": int(row["wins"] or 0),
                "losses": int(row["losses"] or 0),
                "win_rate": _round(row["win_rate"]),
                "profit": _round(row["profit"]),
                "roi": _round(row["roi"]),
                "avg_confidence": _round(row["avg_confidence"]),
                "avg_edge": _round(row["avg_edge"]),
                "avg_elo_diff": _round(row["avg_elo_diff"], 1),
                "source": "pattern_stats",
            }

    for name in ordered:
        if name not in _COHORT_SQL:
            continue
        computed = _aggregate_sql(conn, _COHORT_SQL[name])
        if name not in stats:
            stats[name] = {"name": name, "description": name.replace("_", " "), **computed, "source": "backtest_fight_pool"}
        else:
            stats[name]["computed"] = computed

    return stats


def _similar_rows(conn: sqlite3.Connection | None, frame: dict[str, Any], current: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    if conn is None or not _table_exists(conn, "backtest_fight_pool"):
        return []
    model = frame["model_context"]
    rows = conn.execute(
        """
        SELECT
            date, fighter1, fighter2, pick, pick_prob, pick_odds, edge, pick_elo_diff,
            pick_correct, actual_pnl, skip_reason,
            (
                ABS(pick_prob - ?)
                + ABS(COALESCE(pick_odds, 0) - ?) / 600.0
                + ABS(COALESCE(pick_elo_diff, 0) - ?) / 300.0
                + ABS(COALESCE(edge, 0) - ?) / 0.5
            ) AS distance
        FROM backtest_fight_pool
        WHERE pick_correct IS NOT NULL
        ORDER BY distance ASC, date DESC
        LIMIT ?
        """,
        (
            model.get("pick_prob") or 0.5,
            model.get("pick_odds") or 0,
            current.get("pick_elo_diff") or 0,
            model.get("edge") or 0.0,
            int(limit),
        ),
    ).fetchall()
    output = []
    for row in rows:
        data = _row_to_dict(row)
        output.append(
            {
                "date": data["date"],
                "matchup": f"{data['fighter1']} vs {data['fighter2']}",
                "pick": data["pick"],
                "pick_prob": data["pick_prob"],
                "pick_odds": data["pick_odds"],
                "edge": data["edge"],
                "pick_elo_diff": data["pick_elo_diff"],
                "pick_correct": data["pick_correct"],
                "actual_pnl": data["actual_pnl"],
                "skip_reason": data["skip_reason"],
                "distance": _round(data["distance"]),
            }
        )
    return output


def _stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wins = sum(1 for row in rows if row.get("pick_correct") is True or row.get("pick_correct") == 1)
    losses = sum(1 for row in rows if row.get("pick_correct") is False or row.get("pick_correct") == 0)
    profit = sum(float(row.get("actual_pnl") or 0.0) for row in rows)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 4) if n else None,
        "profit": round(profit, 4),
        "roi": round(profit / n, 4) if n else None,
        "avg_confidence": _round(sum(float(row.get("pick_prob") or 0.0) for row in rows) / n if n else None),
        "avg_edge": _round(sum(float(row.get("edge") or 0.0) for row in rows) / n if n else None),
        "avg_elo_diff": _round(sum(float(row.get("pick_elo_diff") or 0.0) for row in rows) / n if n else None, 1),
    }


def _cohort_for_trait_overlay(frame: dict[str, Any], current: dict[str, Any], hits: dict[str, Any]) -> tuple[str, str]:
    labels = hits.get("labels") or []
    if "underdog_elo_against" in labels:
        return "plus_money_elo_against", "pick_odds > 0 AND pick_elo_diff < 0"
    if "golden_elo_not_expensive" in labels:
        return "golden_elo_not_expensive", _COHORT_SQL["golden_elo_not_expensive"]
    if current.get("pick_elo_diff") is not None and current["pick_elo_diff"] < 0:
        return "model_pick_lower_elo", _COHORT_SQL["model_pick_lower_elo"]
    return "all_graded_with_traits", "1 = 1"


def _trait_overlays(conn: sqlite3.Connection | None, frame: dict[str, Any], current: dict[str, Any], hits: dict[str, Any]) -> list[dict[str, Any]]:
    if conn is None or not (_table_exists(conn, "backtest_fight_pool") and _table_exists(conn, "evidence_items")):
        return []
    cohort_name, cohort_sql = _cohort_for_trait_overlay(frame, current, hits)
    rows = conn.execute(
        f"""
        SELECT p.pick_correct, p.actual_pnl, p.pick_prob, p.edge, p.pick_elo_diff, e.data_json
        FROM backtest_fight_pool p
        JOIN evidence_items e ON e.fight_pool_id = p.id AND e.evidence_type = 'trait_delta'
        WHERE p.pick_correct IS NOT NULL AND ({cohort_sql})
        """
    ).fetchall()

    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["data_json"] or "{}")
        except json.JSONDecodeError:
            continue
        parsed.append({**_row_to_dict(row), "deltas": payload.get("deltas") or {}})

    predicates: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        (
            "primary_support_any",
            "cardio +10, striking efficiency +10, or defensive exposure -10/lower",
            lambda r: (r["deltas"].get("cardio_score_diff") or 0) >= 10
            or (r["deltas"].get("striking_efficiency_score_diff") or 0) >= 10
            or (r["deltas"].get("defensive_exposure_score_diff") is not None and r["deltas"].get("defensive_exposure_score_diff") <= -10),
        ),
        ("cardio_10_plus", "cardio_score_diff >= 10", lambda r: (r["deltas"].get("cardio_score_diff") or 0) >= 10),
        (
            "striking_efficiency_10_plus",
            "striking_efficiency_score_diff >= 10",
            lambda r: (r["deltas"].get("striking_efficiency_score_diff") or 0) >= 10,
        ),
        (
            "safer_defensive_exposure_10_plus",
            "defensive_exposure_score_diff <= -10",
            lambda r: r["deltas"].get("defensive_exposure_score_diff") is not None and r["deltas"].get("defensive_exposure_score_diff") <= -10,
        ),
        ("scramble_10_plus", "scramble_score_diff >= 10", lambda r: (r["deltas"].get("scramble_score_diff") or 0) >= 10),
        (
            "pressure_only_weak_support",
            "striking_pressure_score_diff >= 10 without primary support",
            lambda r: (r["deltas"].get("striking_pressure_score_diff") or 0) >= 10
            and not (
                (r["deltas"].get("cardio_score_diff") or 0) >= 10
                or (r["deltas"].get("striking_efficiency_score_diff") or 0) >= 10
                or (r["deltas"].get("defensive_exposure_score_diff") is not None and r["deltas"].get("defensive_exposure_score_diff") <= -10)
            ),
        ),
        (
            "primary_caution_any",
            "cardio -10, striking efficiency -10, or defensive exposure +10/higher",
            lambda r: (r["deltas"].get("cardio_score_diff") is not None and r["deltas"].get("cardio_score_diff") <= -10)
            or (r["deltas"].get("striking_efficiency_score_diff") is not None and r["deltas"].get("striking_efficiency_score_diff") <= -10)
            or (r["deltas"].get("defensive_exposure_score_diff") or 0) >= 10,
        ),
    ]

    overlays = []
    for name, description, predicate in predicates:
        matching = [row for row in parsed if predicate(row)]
        overlays.append({"name": name, "description": description, "cohort": cohort_name, **_stats_from_rows(matching)})
    return overlays


def build_predict_context(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: date | datetime | None = None,
    fighter1_odds: int | None = None,
    fighter2_odds: int | None = None,
    session=None,
    context_pool_path: Path = DEFAULT_CONTEXT_POOL,
    similar_limit: int = 8,
) -> dict[str, Any]:
    created_session = session is None
    session = session or _Session()
    conn: sqlite3.Connection | None = None
    try:
        frame = build_prediction_frame(
            fighter1=fighter1,
            fighter2=fighter2,
            fight_date=fight_date,
            fighter1_odds=fighter1_odds,
            fighter2_odds=fighter2_odds,
            session=session,
        )
        conn, data_quality = _open_context_pool(context_pool_path)
        current = _extract_current_context(frame, session=session, quality=data_quality)
        hits = _pattern_hits(frame, current)
        historical = _historical_bucket_stats(conn, hits["labels"])
        similar = _similar_rows(conn, frame, current, limit=similar_limit)
        overlays = _trait_overlays(conn, frame, current, hits)
        risks = _risk_flags(frame, current, hits)

        return {
            "request": frame["request"],
            "model_context": frame["model_context"],
            "current_context": current,
            "pattern_hits": hits,
            "historical_bucket_stats": historical,
            "similar_rows": similar,
            "trait_overlays": overlays,
            "risk_flags": risks,
            "data_quality": data_quality,
        }
    finally:
        if conn is not None:
            conn.close()
        if created_session:
            session.close()
