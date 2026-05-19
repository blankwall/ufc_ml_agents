#!/usr/bin/env python3
"""
Build a fight-level identity map between the main UFC ML SQLite database and
the Sergey enrichment sidecar.

The mapper writes `fight_identity_map` into the sidecar DB. It prefers exact
UFCStats fighter URL pairs, then normalized/alias-aware fighter-name pairs, with
exact-date matches preferred over +/- day matches. Ambiguous matches are flagged
instead of silently accepted.
"""

from __future__ import annotations

import argparse
import ast
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_DB = ROOT_DIR / "data" / "ufc_database.db"
DEFAULT_SIDECAR_DB = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
DEFAULT_ALIAS_SOURCES = [
    ROOT_DIR / "fastapi_app" / "services" / "predict_service.py",
    ROOT_DIR / "backtest" / "backtest_2025.py",
]


@dataclass(frozen=True)
class MainFight:
    fight_id: int
    fight_hash: str | None
    event_id: int
    event_date: date
    event_name: str | None
    fighter_1_id: int
    fighter_1_name: str
    fighter_1_url: str | None
    fighter_1_dob: str | None
    fighter_2_id: int
    fighter_2_name: str
    fighter_2_url: str | None
    fighter_2_dob: str | None


@dataclass(frozen=True)
class SergeyFight:
    fight_id: int
    event_id: int | None
    event_date: date | None
    fight_date: date
    event_name: str | None
    fighter_red_id: int
    fighter_red_name: str | None
    fighter_red_url: str | None
    fighter_red_dob: str | None
    fighter_blue_id: int
    fighter_blue_name: str | None
    fighter_blue_url: str | None
    fighter_blue_dob: str | None
    winner_id: int | None
    fighter_red_elo: int | None
    fighter_blue_elo: int | None
    fight_status: str | None
    promotion: str | None


@dataclass(frozen=True)
class Candidate:
    fight: SergeyFight
    method: str
    score: float
    date_delta_days: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build main DB -> Sergey sidecar fight identity mapping")
    parser.add_argument("--main-db", type=Path, default=DEFAULT_MAIN_DB)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR_DB)
    parser.add_argument("--date-tolerance-days", type=int, default=1)
    parser.add_argument(
        "--alias-source",
        type=Path,
        action="append",
        default=[],
        help="Python file containing alias dictionaries to parse. Can be repeated.",
    )
    parser.add_argument(
        "--include-unmatched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write unmatched/ambiguous main fights into fight_identity_map (default: true).",
    )
    return parser.parse_args()


def normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip().lower().rstrip("/")
    if not value:
        return None
    return value.replace("http://", "https://", 1)


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    value = name.strip().lower()
    value = re.sub(r"['’.`]", "", value)
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text[:19] if "%H" in fmt else text, fmt).date()
        except ValueError:
            continue
    return None


def parse_aliases_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            # Handles `VAR: dict[str, str] = {...}` annotated assignments.
            names = [node.target.id] if isinstance(node.target, ast.Name) else []
            value_node = node.value
        else:
            continue
        if not any(name in {"FIGHTER_ALIASES", "_NAME_FIXES"} for name in names):
            continue
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, dict):
            continue
        for raw, canonical in value.items():
            if isinstance(raw, str) and isinstance(canonical, str):
                aliases[normalize_name(raw)] = normalize_name(canonical)
    return aliases


def load_aliases(paths: list[Path]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for path in paths:
        aliases.update(parse_aliases_from_file(path))
    # Make alias resolution idempotent for common already-canonical values.
    for canonical in list(aliases.values()):
        aliases.setdefault(canonical, canonical)
    return aliases


def canonical_name(name: str | None, aliases: dict[str, str]) -> str:
    normalized = normalize_name(name)
    return aliases.get(normalized, normalized)


def pair_key(a: str | None, b: str | None) -> tuple[str, str] | None:
    if not a or not b:
        return None
    return tuple(sorted((a, b)))


def load_main_fights(conn: sqlite3.Connection) -> list[MainFight]:
    rows = conn.execute(
        """
        SELECT
            f.id AS fight_id,
            f.fight_id AS fight_hash,
            e.id AS event_id,
            e.date AS event_date,
            e.name AS event_name,
            f1.id AS fighter_1_id,
            f1.name AS fighter_1_name,
            f1.url AS fighter_1_url,
            f1.date_of_birth AS fighter_1_dob,
            f2.id AS fighter_2_id,
            f2.name AS fighter_2_name,
            f2.url AS fighter_2_url,
            f2.date_of_birth AS fighter_2_dob
        FROM fights f
        JOIN events e ON e.id = f.event_id
        JOIN fighters f1 ON f1.id = f.fighter_1_id
        JOIN fighters f2 ON f2.id = f.fighter_2_id
        ORDER BY f.id
        """
    ).fetchall()
    fights: list[MainFight] = []
    skipped = 0
    for row in rows:
        event_date = parse_date(row["event_date"])
        if event_date is None:
            skipped += 1
            continue
        fights.append(
            MainFight(
                fight_id=row["fight_id"],
                fight_hash=row["fight_hash"],
                event_id=row["event_id"],
                event_date=event_date,
                event_name=row["event_name"],
                fighter_1_id=row["fighter_1_id"],
                fighter_1_name=row["fighter_1_name"],
                fighter_1_url=row["fighter_1_url"],
                fighter_1_dob=row["fighter_1_dob"],
                fighter_2_id=row["fighter_2_id"],
                fighter_2_name=row["fighter_2_name"],
                fighter_2_url=row["fighter_2_url"],
                fighter_2_dob=row["fighter_2_dob"],
            )
        )
    if skipped:
        print(f"Skipped {skipped} main fights with unparseable dates", file=sys.stderr)
    return fights


def load_sergey_fights(conn: sqlite3.Connection) -> list[SergeyFight]:
    rows = conn.execute(
        """
        SELECT
            fight.fight_id,
            fight.event_id,
            fight.event_date,
            fight.fight_date,
            fight.event_name,
            fight.fighter_red_id,
            fight.fighter_red_name,
            red.ufc_stats_url AS fighter_red_url,
            red.dob AS fighter_red_dob,
            fight.fighter_blue_id,
            fight.fighter_blue_name,
            blue.ufc_stats_url AS fighter_blue_url,
            blue.dob AS fighter_blue_dob,
            fight.winner_id,
            fight.fighter_red_elo,
            fight.fighter_blue_elo,
            fight.fight_status,
            fight.promotion
        FROM fights fight
        JOIN fighters red ON red.fighter_id = fight.fighter_red_id
        JOIN fighters blue ON blue.fighter_id = fight.fighter_blue_id
        WHERE fight.fight_date IS NOT NULL
        ORDER BY fight.fight_id
        """
    ).fetchall()
    fights: list[SergeyFight] = []
    skipped = 0
    for row in rows:
        fight_date = parse_date(row["fight_date"])
        if fight_date is None:
            skipped += 1
            continue
        fights.append(
            SergeyFight(
                fight_id=row["fight_id"],
                event_id=row["event_id"],
                event_date=parse_date(row["event_date"]),
                fight_date=fight_date,
                event_name=row["event_name"],
                fighter_red_id=row["fighter_red_id"],
                fighter_red_name=row["fighter_red_name"],
                fighter_red_url=row["fighter_red_url"],
                fighter_red_dob=row["fighter_red_dob"],
                fighter_blue_id=row["fighter_blue_id"],
                fighter_blue_name=row["fighter_blue_name"],
                fighter_blue_url=row["fighter_blue_url"],
                fighter_blue_dob=row["fighter_blue_dob"],
                winner_id=row["winner_id"],
                fighter_red_elo=row["fighter_red_elo"],
                fighter_blue_elo=row["fighter_blue_elo"],
                fight_status=row["fight_status"],
                promotion=row["promotion"],
            )
        )
    if skipped:
        print(f"Skipped {skipped} Sergey fights with unparseable dates", file=sys.stderr)
    return fights


def build_sergey_indexes(
    sergey_fights: list[SergeyFight],
    aliases: dict[str, str],
) -> tuple[dict[tuple[date, tuple[str, str]], list[SergeyFight]], dict[tuple[date, tuple[str, str]], list[SergeyFight]]]:
    by_url: dict[tuple[date, tuple[str, str]], list[SergeyFight]] = defaultdict(list)
    by_name: dict[tuple[date, tuple[str, str]], list[SergeyFight]] = defaultdict(list)

    for fight in sergey_fights:
        url_pair = pair_key(normalize_url(fight.fighter_red_url), normalize_url(fight.fighter_blue_url))
        if url_pair:
            by_url[(fight.fight_date, url_pair)].append(fight)

        name_pair = pair_key(
            canonical_name(fight.fighter_red_name, aliases),
            canonical_name(fight.fighter_blue_name, aliases),
        )
        if name_pair:
            by_name[(fight.fight_date, name_pair)].append(fight)

    return by_url, by_name


def candidate_quality(fight: SergeyFight) -> float:
    score = 0.0
    if fight.fighter_red_elo is not None and fight.fighter_blue_elo is not None:
        score += 0.030
    if fight.event_id is not None:
        score += 0.020
    event_text = " ".join(filter(None, [fight.event_name, fight.promotion])).lower()
    if "ufc" in event_text:
        score += 0.020
    if fight.winner_id is not None:
        score += 0.010
    if fight.fight_status and fight.fight_status.upper() not in {"TBD", "SCHEDULED"}:
        score += 0.005
    return score


def find_candidates(
    main: MainFight,
    by_url: dict[tuple[date, tuple[str, str]], list[SergeyFight]],
    by_name: dict[tuple[date, tuple[str, str]], list[SergeyFight]],
    aliases: dict[str, str],
    tolerance_days: int,
) -> list[Candidate]:
    url_pair = pair_key(normalize_url(main.fighter_1_url), normalize_url(main.fighter_2_url))
    name_pair = pair_key(
        canonical_name(main.fighter_1_name, aliases),
        canonical_name(main.fighter_2_name, aliases),
    )
    candidates: dict[int, Candidate] = {}

    def add_from_index(
        index: dict[tuple[date, tuple[str, str]], list[SergeyFight]],
        pair: tuple[str, str] | None,
        method_prefix: str,
        base_score: float,
    ) -> None:
        if not pair:
            return
        for delta in range(0, tolerance_days + 1):
            offsets = [0] if delta == 0 else [-delta, delta]
            for offset in offsets:
                target_date = main.event_date + timedelta(days=offset)
                for fight in index.get((target_date, pair), []):
                    date_penalty = 0.025 * abs(offset)
                    method = f"{method_prefix}_date_exact" if offset == 0 else f"{method_prefix}_date_near"
                    score = base_score + candidate_quality(fight) - date_penalty
                    existing = candidates.get(fight.fight_id)
                    candidate = Candidate(
                        fight=fight,
                        method=method,
                        score=score,
                        date_delta_days=abs(offset),
                    )
                    if existing is None or candidate.score > existing.score:
                        candidates[fight.fight_id] = candidate

    add_from_index(by_url, url_pair, "url_pair", 0.950)
    add_from_index(by_name, name_pair, "name_pair", 0.850)

    return sorted(candidates.values(), key=lambda c: (-c.score, c.fight.fight_id))


def init_mapping_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS fight_identity_map;

        CREATE TABLE fight_identity_map (
            main_fight_id INTEGER PRIMARY KEY,
            main_fight_hash TEXT,
            main_event_id INTEGER NOT NULL,
            main_event_date TEXT NOT NULL,
            main_event_name TEXT,
            main_fighter_1_id INTEGER NOT NULL,
            main_fighter_1_name TEXT NOT NULL,
            main_fighter_2_id INTEGER NOT NULL,
            main_fighter_2_name TEXT NOT NULL,
            sergey_fight_id INTEGER,
            sergey_event_id INTEGER,
            sergey_fight_date TEXT,
            sergey_event_name TEXT,
            sergey_fighter_red_id INTEGER,
            sergey_fighter_red_name TEXT,
            sergey_fighter_blue_id INTEGER,
            sergey_fighter_blue_name TEXT,
            match_method TEXT,
            match_score REAL NOT NULL,
            candidate_count INTEGER NOT NULL,
            date_delta_days INTEGER,
            review_status TEXT NOT NULL,
            notes TEXT
        );

        CREATE INDEX idx_fight_identity_map_sergey_fight_id ON fight_identity_map(sergey_fight_id);
        CREATE INDEX idx_fight_identity_map_status ON fight_identity_map(review_status);
        CREATE INDEX idx_fight_identity_map_main_date ON fight_identity_map(main_event_date);
        """
    )


def choose_candidate(candidates: list[Candidate]) -> tuple[Candidate | None, str, str | None]:
    if not candidates:
        return None, "unmatched", "no candidate matched by URL or normalized fighter-pair/date"

    best = candidates[0]
    if len(candidates) == 1:
        return best, "auto_matched", None

    second = candidates[1]
    if best.score - second.score >= 0.010:
        return best, "auto_matched", f"selected best of {len(candidates)} candidates"

    return None, "ambiguous", f"{len(candidates)} candidates; top scores too close"


def build_mapping(
    main_conn: sqlite3.Connection,
    sidecar_conn: sqlite3.Connection,
    aliases: dict[str, str],
    tolerance_days: int,
    include_unmatched: bool,
) -> dict[str, int]:
    main_fights = load_main_fights(main_conn)
    sergey_fights = load_sergey_fights(sidecar_conn)
    by_url, by_name = build_sergey_indexes(sergey_fights, aliases)
    init_mapping_table(sidecar_conn)

    stats: dict[str, int] = defaultdict(int)
    rows: list[tuple[Any, ...]] = []
    for main in main_fights:
        candidates = find_candidates(main, by_url, by_name, aliases, tolerance_days)
        chosen, status, notes = choose_candidate(candidates)
        stats[status] += 1

        if chosen is None and not include_unmatched:
            continue

        sergey = chosen.fight if chosen else None
        rows.append(
            (
                main.fight_id,
                main.fight_hash,
                main.event_id,
                main.event_date.isoformat(),
                main.event_name,
                main.fighter_1_id,
                main.fighter_1_name,
                main.fighter_2_id,
                main.fighter_2_name,
                sergey.fight_id if sergey else None,
                sergey.event_id if sergey else None,
                sergey.fight_date.isoformat() if sergey else None,
                sergey.event_name if sergey else None,
                sergey.fighter_red_id if sergey else None,
                sergey.fighter_red_name if sergey else None,
                sergey.fighter_blue_id if sergey else None,
                sergey.fighter_blue_name if sergey else None,
                chosen.method if chosen else None,
                chosen.score if chosen else 0.0,
                len(candidates),
                chosen.date_delta_days if chosen else None,
                status,
                notes,
            )
        )

    sidecar_conn.executemany(
        """
        INSERT INTO fight_identity_map (
            main_fight_id,
            main_fight_hash,
            main_event_id,
            main_event_date,
            main_event_name,
            main_fighter_1_id,
            main_fighter_1_name,
            main_fighter_2_id,
            main_fighter_2_name,
            sergey_fight_id,
            sergey_event_id,
            sergey_fight_date,
            sergey_event_name,
            sergey_fighter_red_id,
            sergey_fighter_red_name,
            sergey_fighter_blue_id,
            sergey_fighter_blue_name,
            match_method,
            match_score,
            candidate_count,
            date_delta_days,
            review_status,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    sidecar_conn.commit()
    stats["main_fights"] = len(main_fights)
    stats["sergey_fights"] = len(sergey_fights)
    stats["rows_written"] = len(rows)
    return stats


def validate_mapping(
    main_conn: sqlite3.Connection,
    sidecar_conn: sqlite3.Connection,
    aliases: dict[str, str],
) -> dict[str, int]:
    main_rows = main_conn.execute(
        """
        SELECT
            fight.id AS fight_id,
            winner.name AS winner_name
        FROM fights fight
        LEFT JOIN fighters winner ON winner.id = fight.winner_id
        """
    ).fetchall()
    main_winners = {
        row["fight_id"]: canonical_name(row["winner_name"], aliases)
        for row in main_rows
        if row["winner_name"]
    }
    sergey_rows = sidecar_conn.execute(
        """
        SELECT
            fight_id,
            winner_name,
            fighter_red_elo,
            fighter_blue_elo
        FROM fights
        """
    ).fetchall()
    sergey = {row["fight_id"]: row for row in sergey_rows}

    stats: dict[str, int] = defaultdict(int)
    rows = sidecar_conn.execute(
        """
        SELECT
            main_fight_id,
            main_fighter_1_name,
            main_fighter_2_name,
            sergey_fight_id,
            sergey_fighter_red_name,
            sergey_fighter_blue_name
        FROM fight_identity_map
        WHERE review_status = 'auto_matched'
        """
    ).fetchall()
    stats["auto_matched"] = len(rows)
    for row in rows:
        main_pair = pair_key(
            canonical_name(row["main_fighter_1_name"], aliases),
            canonical_name(row["main_fighter_2_name"], aliases),
        )
        sergey_pair = pair_key(
            canonical_name(row["sergey_fighter_red_name"], aliases),
            canonical_name(row["sergey_fighter_blue_name"], aliases),
        )
        if main_pair and main_pair == sergey_pair:
            stats["fighter_pair_agreement"] += 1

        sergey_fight = sergey.get(row["sergey_fight_id"])
        if sergey_fight and sergey_fight["fighter_red_elo"] is not None and sergey_fight["fighter_blue_elo"] is not None:
            stats["elo_available"] += 1

        main_winner = main_winners.get(row["main_fight_id"])
        sergey_winner = canonical_name(sergey_fight["winner_name"], aliases) if sergey_fight else ""
        if main_winner and sergey_winner:
            stats["winner_compared"] += 1
            if main_winner == sergey_winner:
                stats["winner_agreement"] += 1

    return stats


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.2f}%"


def print_report(
    conn: sqlite3.Connection,
    stats: dict[str, int],
    validation: dict[str, int],
) -> None:
    print("\nFight identity mapping complete")
    print("=" * 72)
    print(f"Main fights considered:   {stats['main_fights']:,}")
    print(f"Sergey fights available:  {stats['sergey_fights']:,}")
    print(f"Rows written:             {stats['rows_written']:,}")

    print("\nReview status")
    for status, count in conn.execute(
        "SELECT review_status, COUNT(*) FROM fight_identity_map GROUP BY review_status ORDER BY COUNT(*) DESC"
    ):
        print(f"  {status:<14} {count:>6,}")

    print("\nMatch methods")
    for method, count in conn.execute(
        """
        SELECT COALESCE(match_method, 'none') AS method, COUNT(*)
        FROM fight_identity_map
        GROUP BY COALESCE(match_method, 'none')
        ORDER BY COUNT(*) DESC
        """
    ):
        print(f"  {method:<24} {count:>6,}")

    print("\nValidation checks on auto-matched rows")
    auto = validation["auto_matched"]
    print(
        f"  fighter-pair agreement: {validation['fighter_pair_agreement']:,}/{auto:,} "
        f"({_pct(validation['fighter_pair_agreement'], auto)})"
    )
    print(
        f"  ELO available:           {validation['elo_available']:,}/{auto:,} "
        f"({_pct(validation['elo_available'], auto)})"
    )
    compared = validation["winner_compared"]
    print(
        f"  winner agreement:        {validation['winner_agreement']:,}/{compared:,} "
        f"({_pct(validation['winner_agreement'], compared)})"
    )

    print("\nSample unmatched/ambiguous rows")
    sample_rows = conn.execute(
        """
        SELECT main_event_date, main_fighter_1_name, main_fighter_2_name, review_status, candidate_count, notes
        FROM fight_identity_map
        WHERE review_status != 'auto_matched'
        ORDER BY main_event_date DESC
        LIMIT 10
        """
    ).fetchall()
    if not sample_rows:
        print("  none")
        return
    for row in sample_rows:
        print(
            f"  [{row['main_event_date']}] {row['main_fighter_1_name']} vs {row['main_fighter_2_name']} "
            f"-> {row['review_status']} ({row['candidate_count']} candidates; {row['notes']})"
        )


def main() -> None:
    args = parse_args()
    alias_sources = args.alias_source or DEFAULT_ALIAS_SOURCES
    aliases = load_aliases(alias_sources)

    if not args.main_db.exists():
        raise FileNotFoundError(f"Main DB not found: {args.main_db}")
    if not args.sidecar.exists():
        raise FileNotFoundError(f"Sergey sidecar not found: {args.sidecar}")

    main_conn = sqlite3.connect(args.main_db)
    sidecar_conn = sqlite3.connect(args.sidecar)
    main_conn.row_factory = sqlite3.Row
    sidecar_conn.row_factory = sqlite3.Row
    try:
        stats = build_mapping(
            main_conn=main_conn,
            sidecar_conn=sidecar_conn,
            aliases=aliases,
            tolerance_days=args.date_tolerance_days,
            include_unmatched=args.include_unmatched,
        )
        validation = validate_mapping(main_conn, sidecar_conn, aliases)
        print_report(sidecar_conn, stats, validation)
    finally:
        main_conn.close()
        sidecar_conn.close()


if __name__ == "__main__":
    main()
