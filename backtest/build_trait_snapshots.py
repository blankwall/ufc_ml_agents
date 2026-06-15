#!/usr/bin/env python3
"""
Build point-in-time fighter trait snapshots for the independent context sidecar.

This script reads the main UFCStats-derived database plus the Sergey sidecar
identity map and writes a generated SQLite database:

  data/enrichment/trait_snapshots.sqlite

It is evidence-only. Scores are first-pass, interpretable proxies intended for
validation and packet context, not betting rules.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_DB = ROOT_DIR / "data" / "ufc_database.db"
DEFAULT_SIDECAR = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
DEFAULT_OUT = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"
TRAIT_VERSION = "trait_v0_1_stats_totals"


SNAPSHOT_COLUMNS = [
    "main_fight_id",
    "as_of_date",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "sergey_fight_id",
    "sergey_fighter_id",
    "fight_count",
    "wins",
    "losses",
    "draws",
    "recent3_win_rate",
    "recent5_win_rate",
    "finish_win_rate",
    "finish_loss_rate",
    "ko_loss_rate",
    "avg_sig_landed_per_min",
    "avg_sig_absorbed_per_min",
    "avg_sig_attempted_per_min",
    "avg_sig_diff_per_min",
    "avg_striking_accuracy",
    "avg_striking_defense",
    "avg_knockdowns_for",
    "avg_knockdowns_against",
    "avg_takedowns_landed_per_15",
    "avg_takedown_attempts_per_15",
    "avg_takedown_accuracy",
    "avg_takedown_defense",
    "avg_submission_attempts_per_15",
    "avg_control_minutes_per_15",
    "avg_control_conceded_minutes_per_15",
    "avg_control_diff_minutes_per_15",
    "late_fight_sample",
    "late_fight_win_rate",
    "experience_score",
    "recent_form_score",
    "cardio_score",
    "durability_risk_score",
    "defensive_exposure_score",
    "offensive_control_score",
    "anti_control_score",
    "scramble_score",
    "striking_pressure_score",
    "striking_efficiency_score",
    "grappling_threat_score",
    "finishing_threat_score",
    "variance_score",
    "trait_confidence",
    "source_fights_json",
    "trait_inputs_json",
    "trait_version",
    "created_at",
]


@dataclass
class FightSide:
    main_fight_id: int
    fight_date: str
    fighter_id: int
    fighter_name: str
    opponent_id: int
    opponent_name: str
    sergey_fight_id: int | None
    sergey_fighter_id: int | None
    result: str
    method: str | None
    duration_min: float
    reached_round_3: bool
    sig_landed: float
    sig_attempted: float
    sig_absorbed: float
    sig_attempted_against: float
    knockdowns_for: float
    knockdowns_against: float
    takedowns_landed: float
    takedowns_attempted: float
    opponent_takedowns_landed: float
    opponent_takedowns_attempted: float
    submission_attempts: float
    control_seconds: float
    control_conceded_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-db", type=Path, default=DEFAULT_MAIN_DB)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_fraction(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    text = str(value).strip()
    if not text or text == "---":
        return 0.0, 0.0
    if " of " not in text:
        try:
            landed = float(text)
        except ValueError:
            return 0.0, 0.0
        return landed, landed
    left, right = text.split(" of ", 1)
    try:
        return float(left.strip()), float(right.strip())
    except ValueError:
        return 0.0, 0.0


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def parse_control_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text == "---":
        return 0.0
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    mins, secs = text.split(":", 1)
    try:
        return float(int(mins) * 60 + int(secs))
    except ValueError:
        return 0.0


def parse_event_date(value: str) -> str:
    for fmt in ("%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Unsupported event date: {value!r}")


def duration_minutes(round_finished: int | None, time_text: str | None, scheduled_rounds: int | None) -> float:
    scheduled = scheduled_rounds or 3
    if not round_finished:
        return float(scheduled * 5)
    if time_text and ":" in str(time_text):
        mins, secs = str(time_text).split(":", 1)
        try:
            elapsed_in_round = float(int(mins) + int(secs) / 60)
        except ValueError:
            elapsed_in_round = 5.0
    else:
        elapsed_in_round = 5.0
    return max(0.1, float((round_finished - 1) * 5 + elapsed_in_round))


def is_finish(method: str | None) -> bool:
    if not method:
        return False
    text = method.upper()
    return "KO" in text or "TKO" in text or "SUB" in text


def is_ko(method: str | None) -> bool:
    if not method:
        return False
    text = method.upper()
    return "KO" in text or "TKO" in text


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def scale(value: float | None, low: float, high: float) -> float | None:
    if value is None:
        return None
    if high == low:
        return None
    return clamp((value - low) / (high - low) * 100)


def centered(value: float | None, half_range: float) -> float | None:
    if value is None:
        return None
    return clamp(50 + (value / half_range) * 50)


def weighted_mean(parts: list[tuple[float | None, float]]) -> float | None:
    clean = [(value, weight) for value, weight in parts if value is not None and not math.isnan(value)]
    if not clean:
        return None
    return sum(value * weight for value, weight in clean) / sum(weight for _, weight in clean)


def zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else value


def rate_per_15(count: float, duration_min: float) -> float:
    return count / duration_min * 15 if duration_min > 0 else 0.0


def load_sergey_side_map(sidecar_path: Path) -> dict[tuple[int, str], tuple[int | None, int | None]]:
    if not sidecar_path.exists():
        return {}
    conn = sqlite3.connect(sidecar_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                main_fight_id,
                sergey_fight_id,
                main_fighter_1_id,
                main_fighter_2_id,
                sergey_fighter_red_id,
                sergey_fighter_blue_id
            FROM fight_identity_map
            WHERE review_status IN ('auto', 'auto_matched', 'confirmed', 'accepted')
               OR review_status IS NULL
            """
        ).fetchall()
    finally:
        conn.close()

    mapping: dict[tuple[int, str], tuple[int | None, int | None]] = {}
    for row in rows:
        main_fight_id = row["main_fight_id"]
        if main_fight_id is None:
            continue
        mapping[(int(main_fight_id), "fighter_1")] = (row["sergey_fight_id"], row["sergey_fighter_red_id"])
        mapping[(int(main_fight_id), "fighter_2")] = (row["sergey_fight_id"], row["sergey_fighter_blue_id"])
    return mapping


def load_fight_sides(main_db: Path, side_map: dict[tuple[int, str], tuple[int | None, int | None]]) -> list[FightSide]:
    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                f.id AS main_fight_id,
                e.date AS event_date,
                f.fighter_1_id,
                f.fighter_2_id,
                f.winner_id,
                f.result,
                f.method,
                f.round_finished,
                f.time,
                f.scheduled_rounds,
                a.name AS fighter_1_name,
                b.name AS fighter_2_name,
                fs.fighter_1_totals,
                fs.fighter_2_totals,
                fs.significant_strikes
            FROM fights f
            JOIN events e ON e.id = f.event_id
            JOIN fighters a ON a.id = f.fighter_1_id
            JOIN fighters b ON b.id = f.fighter_2_id
            JOIN fight_stats fs ON fs.fight_id = f.id
            WHERE e.date IS NOT NULL
              AND fs.fighter_1_totals IS NOT NULL
              AND fs.fighter_2_totals IS NOT NULL
            ORDER BY e.date, f.id
            """
        ).fetchall()
    finally:
        conn.close()

    sides: list[FightSide] = []
    for row in rows:
        fight_date = parse_event_date(row["event_date"])
        duration = duration_minutes(row["round_finished"], row["time"], row["scheduled_rounds"])
        reached_round_3 = (row["round_finished"] or row["scheduled_rounds"] or 3) >= 3
        totals = {
            "fighter_1": parse_json(row["fighter_1_totals"]),
            "fighter_2": parse_json(row["fighter_2_totals"]),
        }
        sig = parse_json(row["significant_strikes"])
        for side, opponent_side in (("fighter_1", "fighter_2"), ("fighter_2", "fighter_1")):
            fighter_id = int(row[f"{side}_id"])
            opponent_id = int(row[f"{opponent_side}_id"])
            fighter_name = row[f"{side}_name"]
            opponent_name = row[f"{opponent_side}_name"]
            my_totals = totals[side]
            opp_totals = totals[opponent_side]
            my_sig = parse_json(sig.get(side))
            opp_sig = parse_json(sig.get(opponent_side))
            sig_landed, sig_attempted = parse_fraction(my_sig.get("sig_strikes_total") or my_totals.get("sig_strikes"))
            sig_absorbed, sig_attempted_against = parse_fraction(
                opp_sig.get("sig_strikes_total") or opp_totals.get("sig_strikes")
            )
            td_landed, td_attempted = parse_fraction(my_totals.get("takedowns"))
            opp_td_landed, opp_td_attempted = parse_fraction(opp_totals.get("takedowns"))
            sergey_fight_id, sergey_fighter_id = side_map.get((int(row["main_fight_id"]), side), (None, None))
            if row["winner_id"] == fighter_id:
                result = "win"
            elif row["winner_id"] == opponent_id:
                result = "loss"
            elif str(row["result"] or "").lower() == "draw":
                result = "draw"
            else:
                result = "no_contest"
            sides.append(
                FightSide(
                    main_fight_id=int(row["main_fight_id"]),
                    fight_date=fight_date,
                    fighter_id=fighter_id,
                    fighter_name=fighter_name,
                    opponent_id=opponent_id,
                    opponent_name=opponent_name,
                    sergey_fight_id=sergey_fight_id,
                    sergey_fighter_id=sergey_fighter_id,
                    result=result,
                    method=row["method"],
                    duration_min=duration,
                    reached_round_3=reached_round_3,
                    sig_landed=sig_landed,
                    sig_attempted=sig_attempted,
                    sig_absorbed=sig_absorbed,
                    sig_attempted_against=sig_attempted_against,
                    knockdowns_for=parse_number(my_totals.get("knockdowns")),
                    knockdowns_against=parse_number(opp_totals.get("knockdowns")),
                    takedowns_landed=td_landed,
                    takedowns_attempted=td_attempted,
                    opponent_takedowns_landed=opp_td_landed,
                    opponent_takedowns_attempted=opp_td_attempted,
                    submission_attempts=parse_number(my_totals.get("submission_attempts")),
                    control_seconds=parse_control_seconds(my_totals.get("control_time")),
                    control_conceded_seconds=parse_control_seconds(opp_totals.get("control_time")),
                )
            )
    return sides


def compute_snapshot(
    *,
    target: FightSide,
    history: list[FightSide],
    created_at: str,
) -> dict[str, Any]:
    wins = sum(1 for fight in history if fight.result == "win")
    losses = sum(1 for fight in history if fight.result == "loss")
    draws = sum(1 for fight in history if fight.result == "draw")
    graded = wins + losses
    fight_count = len([fight for fight in history if fight.result != "no_contest"])
    recent3 = history[-3:]
    recent5 = history[-5:]
    finish_wins = sum(1 for fight in history if fight.result == "win" and is_finish(fight.method))
    finish_losses = sum(1 for fight in history if fight.result == "loss" and is_finish(fight.method))
    ko_losses = sum(1 for fight in history if fight.result == "loss" and is_ko(fight.method))
    sig_landed_per_min = [fight.sig_landed / fight.duration_min for fight in history if fight.duration_min > 0]
    sig_absorbed_per_min = [fight.sig_absorbed / fight.duration_min for fight in history if fight.duration_min > 0]
    sig_attempted_per_min = [fight.sig_attempted / fight.duration_min for fight in history if fight.duration_min > 0]
    sig_diff_per_min = [
        (fight.sig_landed - fight.sig_absorbed) / fight.duration_min for fight in history if fight.duration_min > 0
    ]
    striking_accuracy = [safe_div(fight.sig_landed, fight.sig_attempted) for fight in history if fight.sig_attempted > 0]
    striking_defense = [
        1 - safe_div(fight.sig_absorbed, fight.sig_attempted_against)
        for fight in history
        if fight.sig_attempted_against > 0
    ]
    takedowns_per_15 = [rate_per_15(fight.takedowns_landed, fight.duration_min) for fight in history]
    takedown_attempts_per_15 = [rate_per_15(fight.takedowns_attempted, fight.duration_min) for fight in history]
    td_accuracy = [safe_div(fight.takedowns_landed, fight.takedowns_attempted) for fight in history if fight.takedowns_attempted > 0]
    td_defense = [
        1 - safe_div(fight.opponent_takedowns_landed, fight.opponent_takedowns_attempted)
        for fight in history
        if fight.opponent_takedowns_attempted > 0
    ]
    sub_attempts_per_15 = [rate_per_15(fight.submission_attempts, fight.duration_min) for fight in history]
    control_min_per_15 = [rate_per_15(fight.control_seconds / 60, fight.duration_min) for fight in history]
    control_conceded_min_per_15 = [
        rate_per_15(fight.control_conceded_seconds / 60, fight.duration_min) for fight in history
    ]
    control_diff_min_per_15 = [
        rate_per_15((fight.control_seconds - fight.control_conceded_seconds) / 60, fight.duration_min)
        for fight in history
    ]
    late_fights = [fight for fight in history if fight.reached_round_3 and fight.result in {"win", "loss"}]

    raw = {
        "recent3_win_rate": safe_div(sum(1 for fight in recent3 if fight.result == "win"), len(recent3)),
        "recent5_win_rate": safe_div(sum(1 for fight in recent5 if fight.result == "win"), len(recent5)),
        "finish_win_rate": safe_div(finish_wins, wins),
        "finish_loss_rate": safe_div(finish_losses, losses),
        "ko_loss_rate": safe_div(ko_losses, losses),
        "avg_sig_landed_per_min": mean(sig_landed_per_min),
        "avg_sig_absorbed_per_min": mean(sig_absorbed_per_min),
        "avg_sig_attempted_per_min": mean(sig_attempted_per_min),
        "avg_sig_diff_per_min": mean(sig_diff_per_min),
        "avg_striking_accuracy": mean(striking_accuracy),
        "avg_striking_defense": mean(striking_defense),
        "avg_knockdowns_for": mean([fight.knockdowns_for for fight in history]),
        "avg_knockdowns_against": mean([fight.knockdowns_against for fight in history]),
        "avg_takedowns_landed_per_15": mean(takedowns_per_15),
        "avg_takedown_attempts_per_15": mean(takedown_attempts_per_15),
        "avg_takedown_accuracy": mean(td_accuracy),
        "avg_takedown_defense": mean(td_defense),
        "avg_submission_attempts_per_15": mean(sub_attempts_per_15),
        "avg_control_minutes_per_15": mean(control_min_per_15),
        "avg_control_conceded_minutes_per_15": mean(control_conceded_min_per_15),
        "avg_control_diff_minutes_per_15": mean(control_diff_min_per_15),
        "late_fight_sample": len(late_fights),
        "late_fight_win_rate": safe_div(sum(1 for fight in late_fights if fight.result == "win"), len(late_fights)),
    }

    experience_score = scale(fight_count, 0, 15)
    recent_form_score = weighted_mean(
        [
            (None if raw["recent3_win_rate"] is None else raw["recent3_win_rate"] * 100, 0.60),
            (centered(mean([(fight.sig_landed - fight.sig_absorbed) / fight.duration_min for fight in recent3]), 3), 0.25),
            (None if raw["finish_win_rate"] is None else raw["finish_win_rate"] * 100, 0.15),
        ]
    )
    cardio_score = None
    if raw["late_fight_sample"] >= 2:
        cardio_score = weighted_mean(
            [
                (None if raw["late_fight_win_rate"] is None else raw["late_fight_win_rate"] * 100, 0.70),
                (centered(raw["avg_control_diff_minutes_per_15"], 5), 0.15),
                (centered(raw["avg_sig_diff_per_min"], 3), 0.15),
            ]
        )
    durability_risk_score = weighted_mean(
        [
            (None if raw["finish_loss_rate"] is None else raw["finish_loss_rate"] * 100, 0.40),
            (None if raw["ko_loss_rate"] is None else raw["ko_loss_rate"] * 100, 0.30),
            (scale(raw["avg_knockdowns_against"], 0, 1.0), 0.20),
            (scale(raw["avg_sig_absorbed_per_min"], 1, 6), 0.10),
        ]
    )
    defensive_exposure_score = weighted_mean(
        [
            (scale(raw["avg_sig_absorbed_per_min"], 1, 7), 0.40),
            (None if raw["avg_striking_defense"] is None else (1 - raw["avg_striking_defense"]) * 100, 0.30),
            (scale(raw["avg_control_conceded_minutes_per_15"], 0, 6), 0.20),
            (scale(raw["avg_knockdowns_against"], 0, 1.0), 0.10),
        ]
    )
    offensive_control_score = weighted_mean(
        [
            (scale(raw["avg_control_minutes_per_15"], 0, 7), 0.45),
            (scale(raw["avg_takedowns_landed_per_15"], 0, 4), 0.35),
            (None if raw["avg_takedown_accuracy"] is None else raw["avg_takedown_accuracy"] * 100, 0.20),
        ]
    )
    anti_control_score = weighted_mean(
        [
            (centered(raw["avg_control_diff_minutes_per_15"], 5), 0.40),
            (None if raw["avg_control_conceded_minutes_per_15"] is None else 100 - scale(raw["avg_control_conceded_minutes_per_15"], 0, 7), 0.25),
            (None if raw["avg_takedown_defense"] is None else raw["avg_takedown_defense"] * 100, 0.20),
            (scale(raw["avg_submission_attempts_per_15"], 0, 2), 0.15),
        ]
    )
    scramble_score = weighted_mean(
        [
            (centered(raw["avg_control_diff_minutes_per_15"], 5), 0.45),
            (scale(raw["avg_submission_attempts_per_15"], 0, 2), 0.30),
            (anti_control_score, 0.25),
        ]
    )
    striking_pressure_score = weighted_mean(
        [
            (scale(raw["avg_sig_landed_per_min"], 1, 7), 0.35),
            (scale(raw["avg_sig_attempted_per_min"], 2, 14), 0.25),
            (centered(raw["avg_sig_diff_per_min"], 3), 0.25),
            (scale(raw["avg_knockdowns_for"], 0, 1.0), 0.15),
        ]
    )
    striking_efficiency_score = weighted_mean(
        [
            (None if raw["avg_striking_accuracy"] is None else raw["avg_striking_accuracy"] * 100, 0.35),
            (None if raw["avg_striking_defense"] is None else raw["avg_striking_defense"] * 100, 0.35),
            (centered(raw["avg_sig_diff_per_min"], 3), 0.30),
        ]
    )
    grappling_threat_score = weighted_mean(
        [
            (offensive_control_score, 0.45),
            (scale(raw["avg_submission_attempts_per_15"], 0, 2), 0.35),
            (scale(raw["avg_takedown_attempts_per_15"], 0, 6), 0.20),
        ]
    )
    finishing_threat_score = weighted_mean(
        [
            (None if raw["finish_win_rate"] is None else raw["finish_win_rate"] * 100, 0.55),
            (scale(raw["avg_knockdowns_for"], 0, 1.0), 0.25),
            (scale(raw["avg_submission_attempts_per_15"], 0, 2), 0.20),
        ]
    )
    variance_score = weighted_mean(
        [
            (None if raw["finish_win_rate"] is None else raw["finish_win_rate"] * 100, 0.30),
            (None if raw["finish_loss_rate"] is None else raw["finish_loss_rate"] * 100, 0.30),
            (scale(zero_if_none(raw["avg_knockdowns_for"]) + zero_if_none(raw["avg_knockdowns_against"]), 0, 1.5), 0.20),
            (None if raw["late_fight_win_rate"] is None else abs(raw["late_fight_win_rate"] - 0.5) * 100, 0.20),
        ]
    )

    source_fights = [
        {
            "fight_id": fight.main_fight_id,
            "date": fight.fight_date,
            "opponent": fight.opponent_name,
            "result": fight.result,
            "method": fight.method,
            "sig_diff": fight.sig_landed - fight.sig_absorbed,
            "control_diff_seconds": fight.control_seconds - fight.control_conceded_seconds,
        }
        for fight in history[-3:]
    ]
    trait_inputs = {
        **raw,
        "scoring_note": "trait_v0_1 uses only prior fight totals; no round_by_round data was available in main DB",
    }
    return {
        "main_fight_id": target.main_fight_id,
        "as_of_date": target.fight_date,
        "fighter_id": target.fighter_id,
        "fighter_name": target.fighter_name,
        "opponent_id": target.opponent_id,
        "opponent_name": target.opponent_name,
        "sergey_fight_id": target.sergey_fight_id,
        "sergey_fighter_id": target.sergey_fighter_id,
        "fight_count": fight_count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        **raw,
        "experience_score": experience_score,
        "recent_form_score": recent_form_score,
        "cardio_score": cardio_score,
        "durability_risk_score": durability_risk_score,
        "defensive_exposure_score": defensive_exposure_score,
        "offensive_control_score": offensive_control_score,
        "anti_control_score": anti_control_score,
        "scramble_score": scramble_score,
        "striking_pressure_score": striking_pressure_score,
        "striking_efficiency_score": striking_efficiency_score,
        "grappling_threat_score": grappling_threat_score,
        "finishing_threat_score": finishing_threat_score,
        "variance_score": variance_score,
        "trait_confidence": clamp(fight_count / 5, 0, 1),
        "source_fights_json": json.dumps(source_fights, sort_keys=True),
        "trait_inputs_json": json.dumps(trait_inputs, sort_keys=True),
        "trait_version": TRAIT_VERSION,
        "created_at": created_at,
    }


def build_snapshots(fight_sides: list[FightSide]) -> list[dict[str, Any]]:
    by_date: dict[str, list[FightSide]] = defaultdict(list)
    for side in fight_sides:
        by_date[side.fight_date].append(side)

    histories: dict[int, list[FightSide]] = defaultdict(list)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    snapshots: list[dict[str, Any]] = []
    for fight_date in sorted(by_date):
        day_sides = sorted(by_date[fight_date], key=lambda item: (item.main_fight_id, item.fighter_id))
        for side in day_sides:
            snapshots.append(compute_snapshot(target=side, history=histories[side.fighter_id], created_at=created_at))
        for side in day_sides:
            histories[side.fighter_id].append(side)
    return snapshots


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP VIEW IF EXISTS v_trait_pair_deltas;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS fighter_trait_snapshots;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE fighter_trait_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_fight_id INTEGER NOT NULL,
            as_of_date TEXT NOT NULL,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            opponent_name TEXT NOT NULL,
            sergey_fight_id INTEGER,
            sergey_fighter_id INTEGER,
            fight_count INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            recent3_win_rate REAL,
            recent5_win_rate REAL,
            finish_win_rate REAL,
            finish_loss_rate REAL,
            ko_loss_rate REAL,
            avg_sig_landed_per_min REAL,
            avg_sig_absorbed_per_min REAL,
            avg_sig_attempted_per_min REAL,
            avg_sig_diff_per_min REAL,
            avg_striking_accuracy REAL,
            avg_striking_defense REAL,
            avg_knockdowns_for REAL,
            avg_knockdowns_against REAL,
            avg_takedowns_landed_per_15 REAL,
            avg_takedown_attempts_per_15 REAL,
            avg_takedown_accuracy REAL,
            avg_takedown_defense REAL,
            avg_submission_attempts_per_15 REAL,
            avg_control_minutes_per_15 REAL,
            avg_control_conceded_minutes_per_15 REAL,
            avg_control_diff_minutes_per_15 REAL,
            late_fight_sample INTEGER NOT NULL,
            late_fight_win_rate REAL,
            experience_score REAL,
            recent_form_score REAL,
            cardio_score REAL,
            durability_risk_score REAL,
            defensive_exposure_score REAL,
            offensive_control_score REAL,
            anti_control_score REAL,
            scramble_score REAL,
            striking_pressure_score REAL,
            striking_efficiency_score REAL,
            grappling_threat_score REAL,
            finishing_threat_score REAL,
            variance_score REAL,
            trait_confidence REAL NOT NULL,
            source_fights_json TEXT NOT NULL,
            trait_inputs_json TEXT NOT NULL,
            trait_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX idx_traits_fight ON fighter_trait_snapshots(main_fight_id);
        CREATE INDEX idx_traits_fighter ON fighter_trait_snapshots(fighter_id, as_of_date);
        CREATE INDEX idx_traits_sergey ON fighter_trait_snapshots(sergey_fight_id, sergey_fighter_id);

        CREATE VIEW v_trait_pair_deltas AS
        SELECT
            a.main_fight_id,
            a.as_of_date,
            a.fighter_id,
            a.fighter_name,
            a.opponent_id,
            a.opponent_name,
            a.fight_count,
            b.fight_count AS opponent_fight_count,
            a.experience_score - b.experience_score AS experience_score_diff,
            a.recent_form_score - b.recent_form_score AS recent_form_score_diff,
            a.cardio_score - b.cardio_score AS cardio_score_diff,
            a.durability_risk_score - b.durability_risk_score AS durability_risk_score_diff,
            a.defensive_exposure_score - b.defensive_exposure_score AS defensive_exposure_score_diff,
            a.offensive_control_score - b.offensive_control_score AS offensive_control_score_diff,
            a.anti_control_score - b.anti_control_score AS anti_control_score_diff,
            a.scramble_score - b.scramble_score AS scramble_score_diff,
            a.striking_pressure_score - b.striking_pressure_score AS striking_pressure_score_diff,
            a.striking_efficiency_score - b.striking_efficiency_score AS striking_efficiency_score_diff,
            a.grappling_threat_score - b.grappling_threat_score AS grappling_threat_score_diff,
            a.finishing_threat_score - b.finishing_threat_score AS finishing_threat_score_diff,
            a.variance_score - b.variance_score AS variance_score_diff,
            a.trait_confidence,
            b.trait_confidence AS opponent_trait_confidence
        FROM fighter_trait_snapshots a
        JOIN fighter_trait_snapshots b
          ON b.main_fight_id = a.main_fight_id
         AND b.fighter_id = a.opponent_id;
        """
    )


def insert_snapshots(conn: sqlite3.Connection, snapshots: list[dict[str, Any]]) -> None:
    placeholders = ", ".join("?" for _ in SNAPSHOT_COLUMNS)
    conn.executemany(
        f"INSERT INTO fighter_trait_snapshots ({', '.join(SNAPSHOT_COLUMNS)}) VALUES ({placeholders})",
        [[row[column] for column in SNAPSHOT_COLUMNS] for row in snapshots],
    )


def insert_metadata(
    conn: sqlite3.Connection,
    *,
    main_db: Path,
    sidecar: Path,
    fight_sides: int,
    snapshots: int,
) -> None:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": "1",
        "trait_version": TRAIT_VERSION,
        "source_main_db": str(main_db.relative_to(ROOT_DIR) if main_db.is_relative_to(ROOT_DIR) else main_db),
        "source_sidecar": str(sidecar.relative_to(ROOT_DIR) if sidecar.is_relative_to(ROOT_DIR) else sidecar),
        "fight_sides_loaded": str(fight_sides),
        "snapshots": str(snapshots),
    }
    conn.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items())


def print_summary(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM fighter_trait_snapshots").fetchone()[0]
    with_prior = conn.execute("SELECT COUNT(*) FROM fighter_trait_snapshots WHERE fight_count > 0").fetchone()[0]
    with_sergey = conn.execute("SELECT COUNT(*) FROM fighter_trait_snapshots WHERE sergey_fighter_id IS NOT NULL").fetchone()[0]
    print(f"Snapshots:              {total}")
    print(f"With prior fights:      {with_prior} ({with_prior / total:.1%})" if total else "With prior fights:      0")
    print(f"With Sergey identity:   {with_sergey} ({with_sergey / total:.1%})" if total else "With Sergey identity:   0")
    print("\nTrait coverage")
    for column in [
        "experience_score",
        "recent_form_score",
        "cardio_score",
        "durability_risk_score",
        "defensive_exposure_score",
        "offensive_control_score",
        "anti_control_score",
        "striking_pressure_score",
        "striking_efficiency_score",
        "grappling_threat_score",
    ]:
        row = conn.execute(
            f"SELECT COUNT(*) AS n, AVG({column}) AS avg_value FROM fighter_trait_snapshots WHERE {column} IS NOT NULL"
        ).fetchone()
        print(f"  {column:<32} n={row['n']:>5} avg={row['avg_value'] if row['avg_value'] is not None else 0:>6.1f}")


def main() -> None:
    args = parse_args()
    if not args.main_db.exists():
        raise SystemExit(f"Main DB not found: {args.main_db}")
    side_map = load_sergey_side_map(args.sidecar)
    fight_sides = load_fight_sides(args.main_db, side_map)
    snapshots = build_snapshots(fight_sides)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    conn = sqlite3.connect(args.out)
    try:
        create_schema(conn)
        insert_snapshots(conn, snapshots)
        insert_metadata(conn, main_db=args.main_db, sidecar=args.sidecar, fight_sides=len(fight_sides), snapshots=len(snapshots))
        conn.commit()
        print(f"Trait snapshots written: {args.out}")
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
