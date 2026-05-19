#!/usr/bin/env python3
"""
Export selected enrichment data from the Sergey Postgres database into a
separate local SQLite sidecar database.

The sidecar keeps the current UFC ML app schema untouched while making
historical ELO snapshots and contextual annotations available for later joins.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import psycopg2
import psycopg2.extras


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"


FIGHTERS_QUERY = """
SELECT
    f.id AS fighter_id,
    f.full_name,
    f.dob::text AS dob,
    f.nickname,
    f.stance,
    f.height,
    f.reach,
    f.weight,
    f.wins,
    f.losses,
    f.draws,
    f.associations,
    f.city,
    f.country,
    f.sherdog_url,
    f.tapology_url,
    f.ufc_stats_url,
    f.elo_current,
    f.elo_peak,
    f.striking_defense,
    f.td_defense,
    f.significant_strikes_landed_per_minute,
    f.significant_strikes_absorbed_per_minute,
    f.significant_striking_accuracy
FROM fighters f
ORDER BY f.id
"""


FIGHTS_QUERY = """
SELECT
    f.id AS fight_id,
    f.event_id,
    e.name AS event_name,
    e.date::text AS event_date,
    f.date::text AS fight_date,
    f.promotion,
    f.division,
    f.fighter_red_id,
    fr.full_name AS fighter_red_name,
    f.fighter_blue_id,
    fb.full_name AS fighter_blue_name,
    f.winner_id,
    w.full_name AS winner_name,
    f.fighter_red_elo,
    f.fighter_blue_elo,
    (f.fighter_red_elo - f.fighter_blue_elo) AS elo_diff,
    f.fight_status::text AS fight_status,
    f.short_method,
    f.scheduled_rounds,
    COALESCE(f.is_main_event, FALSE) AS is_main_event
FROM fights f
JOIN fighters fr ON fr.id = f.fighter_red_id
JOIN fighters fb ON fb.id = f.fighter_blue_id
LEFT JOIN fighters w ON w.id = f.winner_id
LEFT JOIN events e ON e.id = f.event_id
ORDER BY f.id
"""


ASSESSMENTS_QUERY = """
SELECT
    a.id AS assessment_id,
    a.fight_id,
    fight.date::text AS fight_date,
    a.fighter_id,
    fighter.full_name AS fighter_name,
    a.distance_control,
    a.fight_iq,
    a.hittability,
    a.pace_retention,
    a.scramble
FROM assessment a
JOIN fights fight ON fight.id = a.fight_id
JOIN fighters fighter ON fighter.id = a.fighter_id
ORDER BY a.id
"""


DFS_QUERY = """
SELECT
    d.id AS dfs_id,
    d.event_id,
    e.name AS event_name,
    e.date::text AS event_date,
    d.fighter_id,
    f.full_name AS fighter_name,
    d.market_odds,
    d.model_odds,
    d.matchup_style,
    d.opponent_td_defense,
    d.opponent_striking_defense,
    d.performance_trend,
    d.current_streak,
    d.last5_win_rate,
    d.user_notes,
    d.user_win_confidence,
    d.userr1finish_confidence,
    d.userr2finish_confidence,
    d.userr3finish_confidence,
    d.user_volume_multiplier,
    COALESCE(d.is_anchor, FALSE) AS is_anchor,
    COALESCE(d.is_stack_candidate, FALSE) AS is_stack_candidate,
    COALESCE(d.is_faded, FALSE) AS is_faded,
    COALESCE(d.is_main_event, FALSE) AS is_main_event,
    d.scheduled_rounds
FROM dfs_fighter_data d
JOIN fighters f ON f.id = d.fighter_id
JOIN events e ON e.id = d.event_id
ORDER BY d.id
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Sergey Postgres data into a SQLite sidecar DB")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-db", default="ufc-test-1")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output SQLite path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def init_sqlite(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS fighters;
        DROP TABLE IF EXISTS fights;
        DROP TABLE IF EXISTS assessments;
        DROP TABLE IF EXISTS dfs_fighter_data;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE fighters (
            fighter_id INTEGER PRIMARY KEY,
            full_name TEXT,
            dob TEXT,
            nickname TEXT,
            stance TEXT,
            height TEXT,
            reach INTEGER,
            weight INTEGER,
            wins INTEGER,
            losses INTEGER,
            draws INTEGER,
            associations TEXT,
            city TEXT,
            country TEXT,
            sherdog_url TEXT,
            tapology_url TEXT,
            ufc_stats_url TEXT,
            elo_current INTEGER,
            elo_peak INTEGER,
            striking_defense INTEGER,
            td_defense INTEGER,
            significant_strikes_landed_per_minute INTEGER,
            significant_strikes_absorbed_per_minute INTEGER,
            significant_striking_accuracy INTEGER
        );

        CREATE TABLE fights (
            fight_id INTEGER PRIMARY KEY,
            event_id INTEGER,
            event_name TEXT,
            event_date TEXT,
            fight_date TEXT,
            promotion TEXT,
            division TEXT,
            fighter_red_id INTEGER,
            fighter_red_name TEXT,
            fighter_blue_id INTEGER,
            fighter_blue_name TEXT,
            winner_id INTEGER,
            winner_name TEXT,
            fighter_red_elo INTEGER,
            fighter_blue_elo INTEGER,
            elo_diff INTEGER,
            fight_status TEXT,
            short_method TEXT,
            scheduled_rounds INTEGER,
            is_main_event INTEGER NOT NULL
        );

        CREATE TABLE assessments (
            assessment_id INTEGER PRIMARY KEY,
            fight_id INTEGER NOT NULL,
            fight_date TEXT,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT,
            distance_control INTEGER,
            fight_iq INTEGER,
            hittability INTEGER,
            pace_retention INTEGER,
            scramble INTEGER
        );

        CREATE TABLE dfs_fighter_data (
            dfs_id INTEGER PRIMARY KEY,
            event_id INTEGER NOT NULL,
            event_name TEXT,
            event_date TEXT,
            fighter_id INTEGER NOT NULL,
            fighter_name TEXT,
            market_odds REAL,
            model_odds REAL,
            matchup_style TEXT,
            opponent_td_defense REAL,
            opponent_striking_defense REAL,
            performance_trend REAL,
            current_streak INTEGER,
            last5_win_rate REAL,
            user_notes TEXT,
            user_win_confidence REAL,
            userr1finish_confidence REAL,
            userr2finish_confidence REAL,
            userr3finish_confidence REAL,
            user_volume_multiplier REAL,
            is_anchor INTEGER NOT NULL,
            is_stack_candidate INTEGER NOT NULL,
            is_faded INTEGER NOT NULL,
            is_main_event INTEGER NOT NULL,
            scheduled_rounds INTEGER
        );

        CREATE INDEX idx_fighters_full_name ON fighters(full_name);
        CREATE INDEX idx_fighters_ufc_stats_url ON fighters(ufc_stats_url);
        CREATE INDEX idx_fights_date ON fights(fight_date);
        CREATE INDEX idx_fights_pair_date ON fights(fight_date, fighter_red_name, fighter_blue_name);
        CREATE INDEX idx_assessments_fighter ON assessments(fighter_id);
        CREATE INDEX idx_assessments_fight ON assessments(fight_id);
        CREATE INDEX idx_dfs_event_fighter ON dfs_fighter_data(event_id, fighter_id);
        """
    )


def normalize_row(row: Sequence[object]) -> tuple[object, ...]:
    normalized: list[object] = []
    for value in row:
        if isinstance(value, bool):
            normalized.append(int(value))
        else:
            normalized.append(value)
    return tuple(normalized)


def copy_query(
    pg_conn: psycopg2.extensions.connection,
    sqlite_conn: sqlite3.Connection,
    table_name: str,
    query: str,
    batch_size: int = 5000,
) -> int:
    total = 0
    with pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute(query)
        columns = [desc.name for desc in cursor.description]
        column_sql = ", ".join(quote_identifier(col) for col in columns)
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f"INSERT INTO {quote_identifier(table_name)} ({column_sql}) VALUES ({placeholders})"

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            payload = [normalize_row(tuple(row[col] for col in columns)) for row in rows]
            sqlite_conn.executemany(insert_sql, payload)
            sqlite_conn.commit()
            total += len(payload)
            print(f"{table_name}: {total:,} rows")

    return total


def write_metadata(conn: sqlite3.Connection, items: Iterable[tuple[str, str]]) -> None:
    conn.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", list(items))
    conn.commit()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    pg_conn = psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_db,
        user=args.pg_user,
        password=args.pg_password,
    )
    pg_conn.autocommit = False

    sqlite_conn = sqlite3.connect(args.output)
    try:
        init_sqlite(sqlite_conn)
        counts = {
            "fighters": copy_query(pg_conn, sqlite_conn, "fighters", FIGHTERS_QUERY),
            "fights": copy_query(pg_conn, sqlite_conn, "fights", FIGHTS_QUERY),
            "assessments": copy_query(pg_conn, sqlite_conn, "assessments", ASSESSMENTS_QUERY),
            "dfs_fighter_data": copy_query(pg_conn, sqlite_conn, "dfs_fighter_data", DFS_QUERY),
        }
        write_metadata(
            sqlite_conn,
            [
                ("source_db", args.pg_db),
                ("source_host", args.pg_host),
                ("fighters_rows", str(counts["fighters"])),
                ("fights_rows", str(counts["fights"])),
                ("assessments_rows", str(counts["assessments"])),
                ("dfs_fighter_data_rows", str(counts["dfs_fighter_data"])),
            ],
        )
    finally:
        sqlite_conn.close()
        pg_conn.close()

    print(f"Wrote sidecar DB to {args.output}")


if __name__ == "__main__":
    main()
