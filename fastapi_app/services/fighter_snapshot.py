from __future__ import annotations

import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, joinedload, sessionmaker

from backtest.elo_analysis import DEFAULT_ALIAS_SOURCES, load_aliases, normalize_name
from database.schema import BettingOdds, Event, Fight, Fighter
from fastapi_app.services.fighter_identity import FIGHTER_ALIASES, resolve_fighter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_MAIN_DB_CANDIDATES = (ROOT_DIR / "data" / "ufc_database.db", ROOT_DIR / "ufc_database.db")
SIDECAR_DB = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
TRAITS_DB = ROOT_DIR / "data" / "enrichment" / "trait_snapshots.sqlite"


def _main_db_path() -> Path:
    for candidate in _MAIN_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return _MAIN_DB_CANDIDATES[0]


_ENGINE = create_engine(f"sqlite:///{_main_db_path()}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_ENGINE)


@lru_cache(maxsize=1)
def _sidecar_aliases() -> dict[str, str]:
    return load_aliases(DEFAULT_ALIAS_SOURCES)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_date_any(value: str | datetime | date | None) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    text = str(value).strip()
    if not text:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"Could not parse date: {value!r}") from exc


def _iso_date(value: datetime | None) -> str | None:
    return value.date().isoformat() if value else None


def _display_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100, 1) if value <= 1 else round(value, 1)


def _american_odds_display(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return f"+{value}" if value > 0 else str(value)


def _fight_result_for_fighter(fight: Fight, fighter_id: int) -> str:
    if not fight.result:
        return "N/A"
    is_f1 = fight.fighter_1_id == fighter_id
    pos = 1 if is_f1 else 2
    if fight.result == f"fighter_{pos}":
        return "W"
    if fight.result == "draw":
        return "D"
    if fight.result == "no_contest":
        return "NC"
    return "L"


def _main_fights_as_of(session: Session, fighter_id: int, as_of: datetime | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fights = (
        session.query(Fight)
        .options(joinedload(Fight.fighter_1), joinedload(Fight.fighter_2), joinedload(Fight.event))
        .filter(or_(Fight.fighter_1_id == fighter_id, Fight.fighter_2_id == fighter_id))
        .all()
    )

    rows: list[dict[str, Any]] = []
    for fight in fights:
        event = fight.event
        fight_dt = _parse_date_any(event.date if event else None)
        is_f1 = fight.fighter_1_id == fighter_id
        opponent = fight.fighter_2 if is_f1 else fight.fighter_1
        odds_attr = "fighter_1_odds" if is_f1 else "fighter_2_odds"
        close_row = (
            session.query(BettingOdds)
            .filter_by(fight_id=fight.id, is_closing_line=True)
            .filter(getattr(BettingOdds, odds_attr).isnot(None))
            .first()
        )
        odds_value = getattr(close_row, odds_attr, None) if close_row else None
        rows.append(
            {
                "fight": fight,
                "fight_date": fight_dt,
                "date": event.date if event else None,
                "event": event.name if event else None,
                "opponent": opponent.name if opponent else None,
                "result": _fight_result_for_fighter(fight, fighter_id),
                "method": fight.method,
                "method_detail": fight.method_detail,
                "round": fight.round_finished,
                "time": fight.time,
                "weight_class": fight.weight_class,
                "closing_odds": odds_value,
                "closing_odds_display": _american_odds_display(odds_value),
                "is_title_fight": bool(fight.is_title_fight),
            }
        )

    rows.sort(key=lambda row: (row["fight_date"] or datetime.min, row["fight"].id), reverse=True)
    filtered = [row for row in rows if as_of is None or (row["fight_date"] is not None and row["fight_date"] < as_of)]
    return rows, filtered


def _record_from_results(rows: list[dict[str, Any]]) -> dict[str, int]:
    record = {"wins": 0, "losses": 0, "draws": 0, "no_contests": 0}
    for row in rows:
        result = row["result"]
        if result == "W":
            record["wins"] += 1
        elif result == "L":
            record["losses"] += 1
        elif result == "D":
            record["draws"] += 1
        elif result == "NC":
            record["no_contests"] += 1
    return record


def _streak(results: list[str]) -> dict[str, Any] | None:
    if not results:
        return None
    first = results[0]
    if first not in {"W", "L", "D", "NC"}:
        return None
    length = 0
    for result in results:
        if result != first:
            break
        length += 1
    return {"result": first, "length": length}


def _recent_results_summary(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    recent = rows[:limit]
    results = [row["result"] for row in recent]
    return {
        "window": limit,
        "results": results,
        "record": _record_from_results(recent),
        "streak": _streak(results),
    }


def _wmma_status(filtered_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> tuple[Optional[bool], Optional[str]]:
    for source in (filtered_rows, all_rows):
        for row in source:
            weight_class = row["weight_class"]
            if weight_class:
                return weight_class.startswith("Women's"), weight_class
    return None, None


def _sidecar_name_variants(query_name: str, canonical_name: str) -> set[str]:
    normalized_query = normalize_name(query_name)
    normalized_canonical = normalize_name(canonical_name)
    aliases = _sidecar_aliases()
    variants = {normalized_query, normalized_canonical}
    if normalized_query in aliases:
        variants.add(aliases[normalized_query])
    if normalized_canonical in aliases:
        variants.add(aliases[normalized_canonical])
    variants.update(alias for alias, canonical in aliases.items() if canonical in variants)
    for alias, canonical in FIGHTER_ALIASES.items():
        if canonical == canonical_name or alias == query_name:
            variants.add(normalize_name(alias))
            variants.add(normalize_name(canonical))
    return {variant for variant in variants if variant}


def _resolve_sidecar_identity(conn: sqlite3.Connection, *, fighter: Fighter, query_name: str) -> dict[str, Any] | None:
    mapped = conn.execute(
        """
        SELECT sergey_fighter_id, sergey_name, COUNT(*) AS match_count
        FROM (
            SELECT sergey_fighter_red_id AS sergey_fighter_id, sergey_fighter_red_name AS sergey_name
            FROM fight_identity_map
            WHERE main_fighter_1_id = ? AND sergey_fighter_red_id IS NOT NULL
            UNION ALL
            SELECT sergey_fighter_blue_id AS sergey_fighter_id, sergey_fighter_blue_name AS sergey_name
            FROM fight_identity_map
            WHERE main_fighter_2_id = ? AND sergey_fighter_blue_id IS NOT NULL
        )
        GROUP BY sergey_fighter_id, sergey_name
        ORDER BY match_count DESC, sergey_fighter_id ASC
        """,
        (fighter.id, fighter.id),
    ).fetchall()

    candidates: list[dict[str, Any]] = []
    if mapped:
        variants = _sidecar_name_variants(query_name, fighter.name)
        exact = [row for row in mapped if normalize_name(row["sergey_name"] or "") in variants]
        best_row = exact[0] if exact else mapped[0]
        fighter_row = conn.execute(
            "SELECT * FROM fighters WHERE fighter_id = ?",
            (best_row["sergey_fighter_id"],),
        ).fetchone()
        if fighter_row is not None:
            payload = dict(fighter_row)
            payload.update(
                {
                    "resolution_source": "fight_identity_map",
                    "match_count": best_row["match_count"],
                }
            )
            return payload

    variants = _sidecar_name_variants(query_name, fighter.name)
    rows = conn.execute(
        "SELECT * FROM fighters WHERE full_name IS NOT NULL LIMIT 50000"
    ).fetchall()
    for row in rows:
        row_name = normalize_name(row["full_name"])
        if any(variant and (variant == row_name or variant in row_name or row_name in variant) for variant in variants):
            candidates.append(dict(row))
        if len(candidates) >= 10:
            break
    if not candidates:
        return None

    exact = [row for row in candidates if normalize_name(row["full_name"]) in variants]
    best = exact[0] if len(exact) == 1 else candidates[0]
    best["resolution_source"] = "name_fuzzy"
    best["candidate_count"] = len(candidates)
    return best


def _sidecar_fight_rows(conn: sqlite3.Connection, fighter_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            f.fight_id,
            COALESCE(f.event_date, f.fight_date) AS event_date,
            f.fight_date,
            f.event_name,
            f.fighter_red_id,
            f.fighter_red_name,
            f.fighter_blue_id,
            f.fighter_blue_name,
            f.fighter_red_elo,
            f.fighter_blue_elo,
            f.winner_name,
            f.winner_id,
            f.short_method,
            f.division,
            f.fight_status
        FROM fights f
        WHERE (f.fighter_red_id = ? OR f.fighter_blue_id = ?)
          AND f.promotion LIKE '%Ultimate Fighting%'
        ORDER BY COALESCE(f.event_date, f.fight_date) DESC, f.fight_id DESC
        """,
        (fighter_id, fighter_id),
    ).fetchall()


def _sidecar_history_payload(
    fighter_row: dict[str, Any],
    fight_rows: list[sqlite3.Row],
    *,
    as_of: datetime | None,
    recent_limit: int,
) -> dict[str, Any]:
    fighter_id = int(fighter_row["fighter_id"])
    full_name = fighter_row["full_name"]
    filtered_history: list[dict[str, Any]] = []
    timeline_states: list[int] = []
    next_elo_after_as_of: int | None = None

    chronological_rows = sorted(
        fight_rows,
        key=lambda row: (_parse_date_any(row["event_date"]) or datetime.min, row["fight_id"]),
    )
    if as_of is not None:
        for row in chronological_rows:
            fight_dt = _parse_date_any(row["event_date"])
            if fight_dt is None:
                continue
            if fight_dt >= as_of:
                is_red = row["fighter_red_id"] == fighter_id
                next_elo_after_as_of = row["fighter_red_elo"] if is_red else row["fighter_blue_elo"]
                break

    for row in fight_rows:
        fight_dt = _parse_date_any(row["event_date"])
        if as_of is not None and (fight_dt is None or fight_dt >= as_of):
            continue

        is_red = row["fighter_red_id"] == fighter_id
        fighter_elo = row["fighter_red_elo"] if is_red else row["fighter_blue_elo"]
        opponent_elo = row["fighter_blue_elo"] if is_red else row["fighter_red_elo"]
        opponent_name = row["fighter_blue_name"] if is_red else row["fighter_red_name"]

        if fighter_elo is not None:
            timeline_states.append(fighter_elo)

        winner_id = row["winner_id"]
        winner_name = row["winner_name"]
        if winner_id is None and not winner_name:
            result = "unknown"
        elif winner_id == fighter_id:
            result = "win"
        elif winner_name and normalize_name(winner_name) == normalize_name(full_name):
            result = "win"
        elif winner_name and normalize_name(winner_name) in {
            normalize_name(row["fighter_red_name"]),
            normalize_name(row["fighter_blue_name"]),
        }:
            result = "loss"
        else:
            status = (row["fight_status"] or "").lower()
            if "draw" in status:
                result = "draw"
            elif "no_contest" in status or "no contest" in status:
                result = "no_contest"
            else:
                result = "unknown"

        elo_diff = None
        if fighter_elo is not None and opponent_elo is not None:
            elo_diff = fighter_elo - opponent_elo

        filtered_history.append(
            {
                "fight_id": row["fight_id"],
                "fight_date": _iso_date(fight_dt),
                "event_name": row["event_name"],
                "opponent_name": opponent_name,
                "result": result,
                "method": row["short_method"],
                "division": row["division"],
                "fighter_pre_elo": fighter_elo,
                "opponent_pre_elo": opponent_elo,
                "elo_diff": elo_diff,
            }
        )

    if as_of is None:
        current_elo = fighter_row.get("elo_current")
        current_source = "fighters_current"
        peak_elo = fighter_row.get("elo_peak")
    else:
        if next_elo_after_as_of is not None:
            current_elo = next_elo_after_as_of
            current_source = "next_fight_pre_elo"
        elif fighter_row.get("elo_current") is not None:
            current_elo = fighter_row.get("elo_current")
            current_source = "fighters_current"
        elif filtered_history:
            current_elo = filtered_history[0]["fighter_pre_elo"]
            current_source = "latest_visible_pre_fight_elo"
        else:
            current_elo = None
            current_source = "unavailable"

        state_values = [value for value in timeline_states if value is not None]
        if current_elo is not None:
            state_values.append(current_elo)
        peak_elo = max(state_values) if state_values else current_elo

    decline_from_peak = None
    if peak_elo is not None and current_elo is not None:
        decline_from_peak = peak_elo - current_elo

    return {
        "available": True,
        "fighter_id": fighter_row.get("fighter_id"),
        "resolved_name": fighter_row.get("full_name"),
        "resolution_source": fighter_row.get("resolution_source"),
        "elo_current": current_elo,
        "elo_peak": peak_elo,
        "elo_decline_from_peak": decline_from_peak,
        "elo_current_source": current_source,
        "fights_in_sidecar": len(filtered_history),
        "fights_total_all_time": len(fight_rows),
        "recent_fights": filtered_history[:recent_limit],
        "history": filtered_history,
        "profile": {
            "nickname": fighter_row.get("nickname"),
            "date_of_birth": fighter_row.get("dob"),
            "association": fighter_row.get("associations"),
            "city": fighter_row.get("city"),
            "country": fighter_row.get("country"),
            "stance": fighter_row.get("stance"),
            "height": fighter_row.get("height"),
            "reach": fighter_row.get("reach"),
            "weight": fighter_row.get("weight"),
            "record": {
                "wins": fighter_row.get("wins"),
                "losses": fighter_row.get("losses"),
                "draws": fighter_row.get("draws"),
            },
        },
    }


def _trait_snapshot(fighter_id: int, as_of: datetime | None) -> dict[str, Any] | None:
    if not TRAITS_DB.exists():
        return None

    conn = _readonly_connection(TRAITS_DB)
    try:
        if as_of is None:
            row = conn.execute(
                """
                SELECT *
                FROM fighter_trait_snapshots
                WHERE fighter_id = ?
                ORDER BY as_of_date DESC, snapshot_id DESC
                LIMIT 1
                """,
                (fighter_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT *
                FROM fighter_trait_snapshots
                WHERE fighter_id = ?
                  AND as_of_date <= ?
                ORDER BY as_of_date DESC, snapshot_id DESC
                LIMIT 1
                """,
                (fighter_id, _iso_date(as_of)),
            ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    snapshot = dict(row)
    return {
        "available": True,
        "as_of_date": snapshot["as_of_date"],
        "trait_version": snapshot["trait_version"],
        "trait_confidence": snapshot["trait_confidence"],
        "fight_count": snapshot["fight_count"],
        "recent3_win_rate": snapshot["recent3_win_rate"],
        "recent5_win_rate": snapshot["recent5_win_rate"],
        "finish_win_rate": snapshot["finish_win_rate"],
        "finish_loss_rate": snapshot["finish_loss_rate"],
        "ko_loss_rate": snapshot["ko_loss_rate"],
        "avg_sig_landed_per_min": snapshot["avg_sig_landed_per_min"],
        "avg_sig_absorbed_per_min": snapshot["avg_sig_absorbed_per_min"],
        "avg_sig_diff_per_min": snapshot["avg_sig_diff_per_min"],
        "avg_control_diff_minutes_per_15": snapshot["avg_control_diff_minutes_per_15"],
        "experience_score": snapshot["experience_score"],
        "recent_form_score": snapshot["recent_form_score"],
        "cardio_score": snapshot["cardio_score"],
        "durability_risk_score": snapshot["durability_risk_score"],
        "defensive_exposure_score": snapshot["defensive_exposure_score"],
        "offensive_control_score": snapshot["offensive_control_score"],
        "anti_control_score": snapshot["anti_control_score"],
        "scramble_score": snapshot["scramble_score"],
        "striking_pressure_score": snapshot["striking_pressure_score"],
        "striking_efficiency_score": snapshot["striking_efficiency_score"],
        "grappling_threat_score": snapshot["grappling_threat_score"],
        "finishing_threat_score": snapshot["finishing_threat_score"],
        "variance_score": snapshot["variance_score"],
    }


def build_fighter_snapshot(
    fighter_name: str,
    *,
    as_of: str | datetime | date | None = None,
    recent_elo_fights: int = 2,
    recent_results_limit: int = 5,
    session: Session | None = None,
) -> dict[str, Any]:
    if recent_elo_fights <= 0:
        raise ValueError("recent_elo_fights must be >= 1")
    if recent_results_limit <= 0:
        raise ValueError("recent_results_limit must be >= 1")

    as_of_dt = _parse_date_any(as_of)
    created_session = session is None
    session = session or _Session()

    try:
        fighter = resolve_fighter(session, fighter_name)
        if fighter is None:
            return {
                "query_name": fighter_name,
                "as_of_date": _iso_date(as_of_dt),
                "resolved": False,
                "note": "Fighter not found in main DB.",
            }

        all_rows, filtered_rows = _main_fights_as_of(session, fighter.id, as_of_dt)
        ufc_record = _record_from_results(filtered_rows)
        recent_results = filtered_rows[:recent_results_limit]
        is_wmma, last_weight_class = _wmma_status(filtered_rows, all_rows)
        traits = _trait_snapshot(fighter.id, as_of_dt)

        elo: dict[str, Any] = {"available": False, "recent_fights": [], "history": []}
        if SIDECAR_DB.exists():
            conn = _readonly_connection(SIDECAR_DB)
            try:
                sidecar_fighter = _resolve_sidecar_identity(conn, fighter=fighter, query_name=fighter_name)
                if sidecar_fighter is not None:
                    elo = _sidecar_history_payload(
                        sidecar_fighter,
                        _sidecar_fight_rows(conn, int(sidecar_fighter["fighter_id"])),
                        as_of=as_of_dt,
                        recent_limit=recent_elo_fights,
                    )
            finally:
                conn.close()

        return {
            "query_name": fighter_name,
            "as_of_date": _iso_date(as_of_dt),
            "resolved": True,
            "identity": {
                "resolved_name": fighter.name,
                "main_fighter_id": fighter.id,
                "fighter_id": fighter.fighter_id,
                "alias_input": fighter_name if fighter_name != fighter.name else None,
                "canonical_name": fighter.name,
                "sidecar_name": elo.get("resolved_name"),
                "sidecar_fighter_id": elo.get("fighter_id"),
            },
            "record": {
                "overall": {
                    "wins": fighter.wins,
                    "losses": fighter.losses,
                    "draws": fighter.draws,
                    "no_contests": fighter.no_contests or 0,
                },
                "ufc_as_of": ufc_record,
                "fight_count_as_of": len(filtered_rows),
                "fight_count_total": len(all_rows),
            },
            "profile": {
                "nickname": fighter.nickname or elo.get("profile", {}).get("nickname"),
                "age": fighter.age,
                "date_of_birth": fighter.date_of_birth or elo.get("profile", {}).get("date_of_birth"),
                "stance": fighter.stance or elo.get("profile", {}).get("stance"),
                "height_cm": fighter.height_cm,
                "height_display": elo.get("profile", {}).get("height"),
                "weight_lbs": fighter.weight_lbs,
                "reach_inches": fighter.reach_inches,
                "association": elo.get("profile", {}).get("association"),
                "city": elo.get("profile", {}).get("city"),
                "country": elo.get("profile", {}).get("country"),
                "is_wmma": is_wmma,
                "last_known_weight_class": last_weight_class,
            },
            "stats": {
                "sig_strikes_landed_per_min": fighter.sig_strikes_landed_per_min,
                "striking_accuracy": fighter.striking_accuracy,
                "striking_accuracy_pct": _display_pct(fighter.striking_accuracy),
                "sig_strikes_absorbed_per_min": fighter.sig_strikes_absorbed_per_min,
                "striking_defense": fighter.striking_defense,
                "striking_defense_pct": _display_pct(fighter.striking_defense),
                "takedown_avg_per_15min": fighter.takedown_avg_per_15min,
                "takedown_accuracy": fighter.takedown_accuracy,
                "takedown_accuracy_pct": _display_pct(fighter.takedown_accuracy),
                "takedown_defense": fighter.takedown_defense,
                "takedown_defense_pct": _display_pct(fighter.takedown_defense),
                "submission_avg_per_15min": fighter.submission_avg_per_15min,
            },
            "elo": {key: value for key, value in elo.items() if key != "history"},
            "recent_results_summary": _recent_results_summary(filtered_rows, recent_results_limit),
            "recent_results": [
                {
                    key: row[key]
                    for key in (
                        "date",
                        "event",
                        "opponent",
                        "result",
                        "method",
                        "method_detail",
                        "round",
                        "time",
                        "weight_class",
                        "closing_odds",
                        "closing_odds_display",
                        "is_title_fight",
                    )
                }
                for row in recent_results
            ],
            "qualitative": traits or {"available": False},
        }
    finally:
        if created_session:
            session.close()
