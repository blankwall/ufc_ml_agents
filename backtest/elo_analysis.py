#!/usr/bin/env python3
"""
Analyze backtest results through Sergey sidecar pre-fight ELO.

This is intentionally a post-processing layer. It does not rerun model
inference; it joins an existing backtest results CSV to
data/enrichment/sergey_sidecar.sqlite and asks ELO-specific questions:

  - Did the model pick agree with the higher-ELO fighter?
  - Do low-confidence model picks perform better when ELO supports them?
  - Is ELO alone informative by magnitude of ELO gap?
  - Are current bet/skip decisions hiding ELO-supported upgrade candidates?

Join strategy:
  1. Prefer results CSV main_fight_id -> fight_identity_map.
  2. Fall back to event date + alias-normalized fighter pair for legacy CSVs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SIDECAR = ROOT_DIR / "data" / "enrichment" / "sergey_sidecar.sqlite"
DEFAULT_ALIAS_SOURCES = [
    ROOT_DIR / "fastapi_app" / "services" / "predict_service.py",
    ROOT_DIR / "backtest" / "backtest_2025.py",
]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)

# Sparse cross-source aliases not currently needed by the prediction path but
# useful when joining historical odds/result rows to Sergey names.
EXTRA_ALIASES = {
    "patricio freire": "patricio pitbull",
    "patricio pitbull": "patricio pitbull",
    "song yadong": "song yadong",
    "yadong song": "song yadong",
    "kim sang wook": "sangwook kim",
    "sang wook kim": "sangwook kim",
    "sangwook kim": "sangwook kim",
    "yizha": "yizha",
    "zha yi": "yizha",
    "benoit saint denis": "benoit saint denis",
    "benoit st denis": "benoit saint denis",
    "zach reese": "zach reese",
    "zachary reese": "zach reese",
    "jose henrique": "jose henrique",
    "jose henrique souza": "jose henrique",
    "suyoung you": "suyoung you",
    "su young you": "suyoung you",
    "loma lookboonmee": "loma lookboonmee",
    "loma petchpromlob": "loma lookboonmee",
    # Sergey sidecar / alternate-source name for the same fighter.
    "konklak suphisara": "loma lookboonmee",
}


@dataclass(frozen=True)
class EloMapEntry:
    main_fight_id: int
    main_event_date: str
    main_fighter_1_name: str
    main_fighter_2_name: str
    sergey_fight_id: int
    sergey_fighter_red_name: str
    sergey_fighter_blue_name: str
    fighter_red_elo: int | None
    fighter_blue_elo: int | None
    winner_name: str | None
    match_method: str | None
    review_status: str


@dataclass
class EnrichedFight:
    row_num: int
    source_row_key: str
    date: str
    main_fight_id: int | None
    fighter1: str
    fighter2: str
    pick: str
    winner: str | None
    pick_prob: float
    pick_odds: int | None
    pick_correct: bool | None
    actual_pnl: float | None
    bet: bool
    skip_reason: str
    female: bool
    edge: float | None
    odds_source_file: str | None
    odds_source_line: int | None
    odds_source_type: str | None
    odds_source_row: str | None
    source_event_id: str | None
    source_url: str | None
    scraped_at: str | None
    bookmaker: str | None
    odds_timestamp: str | None
    odds_is_opening_line: bool | None
    odds_is_closing_line: bool | None
    join_status: str
    join_method: str | None
    sergey_fight_id: int | None
    fighter1_elo: int | None
    fighter2_elo: int | None
    pick_elo: int | None
    opponent_elo: int | None
    pick_elo_diff: int | None
    abs_elo_diff: int | None
    model_agrees_with_elo: bool | None
    elo_pick: str | None
    elo_pick_odds: int | None
    elo_pick_correct: bool | None
    elo_pick_pnl: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process backtest results with Sergey sidecar ELO analysis."
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Backtest results CSV, e.g. backtest/backtest_2026_results.csv",
    )
    parser.add_argument(
        "--sidecar",
        default=DEFAULT_SIDECAR,
        type=Path,
        help=f"Sergey sidecar SQLite DB (default: {DEFAULT_SIDECAR})",
    )
    parser.add_argument(
        "--section",
        choices=["coverage", "pick_diff", "agreement", "low_confidence", "pure_elo", "rules", "all"],
        default="all",
        help="Analysis section to print (default: all)",
    )
    parser.add_argument(
        "--write-enriched",
        type=Path,
        default=None,
        help="Optional path to write an enriched row-level CSV.",
    )
    parser.add_argument(
        "--only-bets",
        action="store_true",
        help="Analyze only rows where bet=True.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include unmatched rows in enriched CSV output.",
    )
    parser.add_argument(
        "--date-tolerance-days",
        type=int,
        default=1,
        help="Fallback date tolerance for legacy CSVs without main_fight_id (default: 1).",
    )
    return parser.parse_args()


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    value = value.strip().lower()
    value = re.sub(r"['’.`]", "", value)
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_aliases_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    aliases: dict[str, str] = {}
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
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
        except (SyntaxError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        for raw, canonical in value.items():
            if isinstance(raw, str) and isinstance(canonical, str):
                aliases[normalize_name(raw)] = normalize_name(canonical)
    return aliases


def load_aliases(paths: Iterable[Path]) -> dict[str, str]:
    aliases = dict(EXTRA_ALIASES)
    for path in paths:
        aliases.update(parse_aliases_from_file(path))
    for canonical in list(aliases.values()):
        aliases.setdefault(canonical, canonical)
    return aliases


def canonical_name(name: str | None, aliases: dict[str, str]) -> str:
    normalized = normalize_name(name)
    return aliases.get(normalized, normalized)


def pair_key(fighter1: str, fighter2: str, aliases: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((canonical_name(fighter1, aliases), canonical_name(fighter2, aliases))))


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text[:19] if "%H" in fmt else text, fmt)
        except ValueError:
            continue
    return None


def american_implied_prob(odds: int | None) -> float | None:
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return None


def pnl_for_odds(won: bool, odds: int | None) -> float | None:
    if odds is None:
        return None
    if not won:
        return -1.0
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def names_match(a: str | None, b: str | None, aliases: dict[str, str]) -> bool:
    return canonical_name(a, aliases) == canonical_name(b, aliases)


def load_elo_map(
    sidecar_path: Path,
    aliases: dict[str, str],
) -> tuple[dict[int, EloMapEntry], dict[tuple[str, tuple[str, str]], list[EloMapEntry]]]:
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar DB not found: {sidecar_path}")

    conn = sqlite3.connect(sidecar_path)
    conn.row_factory = sqlite3.Row
    try:
        mapped_rows = conn.execute(
            """
            SELECT
                m.main_fight_id,
                m.main_event_date,
                m.main_fighter_1_name,
                m.main_fighter_2_name,
                m.sergey_fight_id,
                m.sergey_fighter_red_name,
                m.sergey_fighter_blue_name,
                m.match_method,
                m.review_status,
                f.fighter_red_elo,
                f.fighter_blue_elo,
                f.winner_name
            FROM fight_identity_map m
            JOIN fights f ON f.fight_id = m.sergey_fight_id
            WHERE m.review_status = 'auto_matched'
              AND m.sergey_fight_id IS NOT NULL
            """
        ).fetchall()
        raw_rows = conn.execute(
            """
            SELECT
                fight_id,
                fight_date,
                event_name,
                fighter_red_name,
                fighter_blue_name,
                fighter_red_elo,
                fighter_blue_elo,
                winner_name
            FROM fights
            WHERE fight_date IS NOT NULL
              AND fighter_red_name IS NOT NULL
              AND fighter_blue_name IS NOT NULL
              AND promotion LIKE '%Ultimate Fighting%'
            """
        ).fetchall()
    finally:
        conn.close()

    by_main_id: dict[int, EloMapEntry] = {}
    by_date_pair: dict[tuple[str, tuple[str, str]], list[EloMapEntry]] = defaultdict(list)

    for row in mapped_rows:
        entry = EloMapEntry(
            main_fight_id=int(row["main_fight_id"]),
            main_event_date=str(row["main_event_date"]),
            main_fighter_1_name=str(row["main_fighter_1_name"]),
            main_fighter_2_name=str(row["main_fighter_2_name"]),
            sergey_fight_id=int(row["sergey_fight_id"]),
            sergey_fighter_red_name=str(row["sergey_fighter_red_name"]),
            sergey_fighter_blue_name=str(row["sergey_fighter_blue_name"]),
            fighter_red_elo=parse_int(row["fighter_red_elo"]),
            fighter_blue_elo=parse_int(row["fighter_blue_elo"]),
            winner_name=row["winner_name"],
            match_method=row["match_method"],
            review_status=row["review_status"],
        )
        by_main_id[entry.main_fight_id] = entry

        keys = {
            pair_key(entry.main_fighter_1_name, entry.main_fighter_2_name, aliases),
            pair_key(entry.sergey_fighter_red_name, entry.sergey_fighter_blue_name, aliases),
        }
        for key in keys:
            by_date_pair[(entry.main_event_date, key)].append(entry)

    existing_sergey_ids = {entry.sergey_fight_id for entry in by_main_id.values()}
    for row in raw_rows:
        sergey_fight_id = int(row["fight_id"])
        if sergey_fight_id in existing_sergey_ids:
            continue
        fight_date = str(row["fight_date"])
        red_name = str(row["fighter_red_name"])
        blue_name = str(row["fighter_blue_name"])
        entry = EloMapEntry(
            main_fight_id=-sergey_fight_id,
            main_event_date=fight_date,
            main_fighter_1_name=red_name,
            main_fighter_2_name=blue_name,
            sergey_fight_id=sergey_fight_id,
            sergey_fighter_red_name=red_name,
            sergey_fighter_blue_name=blue_name,
            fighter_red_elo=parse_int(row["fighter_red_elo"]),
            fighter_blue_elo=parse_int(row["fighter_blue_elo"]),
            winner_name=row["winner_name"],
            match_method="sergey_raw_date_pair",
            review_status="raw_sergey",
        )
        by_date_pair[(fight_date, pair_key(red_name, blue_name, aliases))].append(entry)

    return by_main_id, by_date_pair


def elo_for_name(entry: EloMapEntry, name: str, aliases: dict[str, str]) -> int | None:
    target = canonical_name(name, aliases)
    red_names = {
        canonical_name(entry.sergey_fighter_red_name, aliases),
        canonical_name(entry.main_fighter_1_name, aliases),
    }
    blue_names = {
        canonical_name(entry.sergey_fighter_blue_name, aliases),
        canonical_name(entry.main_fighter_2_name, aliases),
    }

    if target == canonical_name(entry.sergey_fighter_red_name, aliases):
        return entry.fighter_red_elo
    if target == canonical_name(entry.sergey_fighter_blue_name, aliases):
        return entry.fighter_blue_elo

    # Fall back to main DB names only when they orient cleanly to Sergey names.
    main1 = canonical_name(entry.main_fighter_1_name, aliases)
    main2 = canonical_name(entry.main_fighter_2_name, aliases)
    red = canonical_name(entry.sergey_fighter_red_name, aliases)
    blue = canonical_name(entry.sergey_fighter_blue_name, aliases)
    if target == main1 and main1 in {red, blue}:
        return entry.fighter_red_elo if main1 == red else entry.fighter_blue_elo
    if target == main2 and main2 in {red, blue}:
        return entry.fighter_red_elo if main2 == red else entry.fighter_blue_elo

    # Defensive fallback for exact-or-aliased name sets. This should rarely be
    # needed, but helps when source order differs and aliases are complete.
    if target in red_names and target not in blue_names:
        return entry.fighter_red_elo
    if target in blue_names and target not in red_names:
        return entry.fighter_blue_elo
    return None


def find_map_entry(
    row: dict[str, str],
    by_main_id: dict[int, EloMapEntry],
    by_date_pair: dict[tuple[str, tuple[str, str]], list[EloMapEntry]],
    aliases: dict[str, str],
    date_tolerance_days: int,
) -> tuple[EloMapEntry | None, str, str | None]:
    main_fight_id = parse_int(row.get("main_fight_id"))
    missing_main_id = False
    if main_fight_id is not None:
        entry = by_main_id.get(main_fight_id)
        if entry:
            return entry, "matched", "main_fight_id"
        missing_main_id = True

    date = str(row.get("date", "")).strip()
    if not date:
        return None, "unmatched", "main_fight_id_missing_in_map" if missing_main_id else "missing_date"
    key = (date, pair_key(row.get("fighter1", ""), row.get("fighter2", ""), aliases))
    candidates = by_date_pair.get(key, [])
    if len(candidates) == 1:
        return candidates[0], "matched", "date_pair"
    if len(candidates) > 1:
        return None, "ambiguous", "date_pair"

    parsed_date = parse_date(date)
    if parsed_date and date_tolerance_days > 0:
        pair = key[1]
        tolerant_candidates: list[EloMapEntry] = []
        for offset in range(1, date_tolerance_days + 1):
            for direction in (-1, 1):
                nearby = (parsed_date + timedelta(days=offset * direction)).strftime("%Y-%m-%d")
                tolerant_candidates.extend(by_date_pair.get((nearby, pair), []))
        unique = {candidate.sergey_fight_id: candidate for candidate in tolerant_candidates}
        if len(unique) == 1:
            return next(iter(unique.values())), "matched", "date_pair_tolerant"
        if len(unique) > 1:
            return None, "ambiguous", "date_pair_tolerant"
    return None, "unmatched", "main_fight_id_missing_in_map" if missing_main_id else "date_pair"


def enrich_results(
    results_path: Path,
    sidecar_path: Path,
    aliases: dict[str, str],
    *,
    date_tolerance_days: int,
) -> list[EnrichedFight]:
    by_main_id, by_date_pair = load_elo_map(sidecar_path, aliases)
    enriched: list[EnrichedFight] = []

    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            if parse_bool(row.get("error")) is True or not row.get("pick"):
                continue

            entry, join_status, join_method = find_map_entry(
                row,
                by_main_id,
                by_date_pair,
                aliases,
                date_tolerance_days,
            )
            f1 = row.get("fighter1", "")
            f2 = row.get("fighter2", "")
            pick = row.get("pick", "")
            winner = row.get("winner") or None
            pick_prob = parse_float(row.get("pick_prob")) or 0.0
            pick_odds = parse_int(row.get("pick_odds"))
            odds1 = parse_int(row.get("odds1"))
            odds2 = parse_int(row.get("odds2"))
            pick_correct = parse_bool(row.get("pick_correct"))
            actual_pnl = parse_float(row.get("actual_pnl"))
            bet = parse_bool(row.get("bet")) is True
            female = parse_bool(row.get("female")) is True
            implied = american_implied_prob(pick_odds)
            edge = pick_prob - implied if implied is not None else None

            f1_elo = f2_elo = pick_elo = opponent_elo = None
            pick_elo_diff = abs_elo_diff = None
            agrees = None
            elo_pick = elo_pick_odds = elo_pick_correct = elo_pick_pnl = None
            sergey_fight_id = None

            if entry:
                sergey_fight_id = entry.sergey_fight_id
                f1_elo = elo_for_name(entry, f1, aliases)
                f2_elo = elo_for_name(entry, f2, aliases)
                if f1_elo is None or f2_elo is None:
                    if entry.fighter_red_elo is None or entry.fighter_blue_elo is None:
                        join_status = "matched_no_elo"
                    else:
                        join_status = "matched_no_orientation"
                else:
                    if names_match(pick, f1, aliases):
                        pick_elo, opponent_elo = f1_elo, f2_elo
                    elif names_match(pick, f2, aliases):
                        pick_elo, opponent_elo = f2_elo, f1_elo
                    if pick_elo is not None and opponent_elo is not None:
                        pick_elo_diff = pick_elo - opponent_elo
                        abs_elo_diff = abs(pick_elo_diff)
                        agrees = pick_elo_diff > 0 if pick_elo_diff != 0 else None

                    if f1_elo != f2_elo:
                        if f1_elo > f2_elo:
                            elo_pick = f1
                            elo_pick_odds = odds1
                        else:
                            elo_pick = f2
                            elo_pick_odds = odds2
                        if winner:
                            elo_pick_correct = names_match(elo_pick, winner, aliases)
                            elo_pick_pnl = pnl_for_odds(elo_pick_correct, elo_pick_odds)

            enriched.append(
                EnrichedFight(
                    row_num=row_num,
                    source_row_key=f"{display_path(results_path)}:{row_num}",
                    date=str(row.get("date", "")).strip(),
                    main_fight_id=parse_int(row.get("main_fight_id")),
                    fighter1=f1,
                    fighter2=f2,
                    pick=pick,
                    winner=winner,
                    pick_prob=pick_prob,
                    pick_odds=pick_odds,
                    pick_correct=pick_correct,
                    actual_pnl=actual_pnl,
                    bet=bet,
                    skip_reason=str(row.get("skip_reason", "") or "").strip(),
                    female=female,
                    edge=edge,
                    odds_source_file=row.get("odds_source_file") or None,
                    odds_source_line=parse_int(row.get("odds_source_line")),
                    odds_source_type=row.get("odds_source_type") or None,
                    odds_source_row=row.get("odds_source_row") or None,
                    source_event_id=row.get("source_event_id") or None,
                    source_url=row.get("source_url") or None,
                    scraped_at=row.get("scraped_at") or None,
                    bookmaker=row.get("bookmaker") or None,
                    odds_timestamp=row.get("odds_timestamp") or None,
                    odds_is_opening_line=parse_bool(row.get("odds_is_opening_line")),
                    odds_is_closing_line=parse_bool(row.get("odds_is_closing_line")),
                    join_status=join_status,
                    join_method=join_method,
                    sergey_fight_id=sergey_fight_id,
                    fighter1_elo=f1_elo,
                    fighter2_elo=f2_elo,
                    pick_elo=pick_elo,
                    opponent_elo=opponent_elo,
                    pick_elo_diff=pick_elo_diff,
                    abs_elo_diff=abs_elo_diff,
                    model_agrees_with_elo=agrees,
                    elo_pick=elo_pick,
                    elo_pick_odds=elo_pick_odds,
                    elo_pick_correct=elo_pick_correct,
                    elo_pick_pnl=elo_pick_pnl,
                )
            )
    return enriched


def analysis_rows(rows: Iterable[EnrichedFight]) -> list[EnrichedFight]:
    return [
        row for row in rows
        if row.join_status == "matched"
        and row.pick_elo_diff is not None
        and row.pick_correct is not None
    ]


def stat_tuple(rows: list[EnrichedFight], *, pnl_attr: str = "actual_pnl", correct_attr: str = "pick_correct") -> tuple:
    count = len(rows)
    if not rows:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    wins = sum(1 for row in rows if getattr(row, correct_attr) is True)
    losses = sum(1 for row in rows if getattr(row, correct_attr) is False)
    graded = wins + losses
    win_rate = wins / graded if graded else 0.0
    profit = sum(float(getattr(row, pnl_attr) or 0.0) for row in rows)
    roi = profit / count * 100 if count else 0.0
    avg_conf = sum(row.pick_prob for row in rows) / count
    avg_edge = sum(row.edge or 0.0 for row in rows) / count
    avg_elo = sum(row.pick_elo_diff or 0 for row in rows) / count
    return count, wins, losses, win_rate, profit, roi, avg_conf, avg_edge, avg_elo


def print_stats_header(label_width: int = 28) -> None:
    print(
        f"{'Segment':<{label_width}} {'N':>5} {'W':>4} {'L':>4} "
        f"{'WinRate':>8} {'Profit':>8} {'ROI':>7} {'AvgConf':>8} {'AvgEdge':>8} {'AvgELO':>8}"
    )
    print("-" * (label_width + 76))


def print_stats_row(label: str, rows: list[EnrichedFight], label_width: int = 28, *, pnl_attr: str = "actual_pnl", correct_attr: str = "pick_correct") -> None:
    n, w, l, wr, profit, roi, conf, edge, elo = stat_tuple(rows, pnl_attr=pnl_attr, correct_attr=correct_attr)
    if n == 0:
        print(f"{label:<{label_width}} {'--':>5} {'--':>4} {'--':>4} {'--':>8} {'--':>8} {'--':>7} {'--':>8} {'--':>8} {'--':>8}")
        return
    print(
        f"{label:<{label_width}} {n:>5} {w:>4} {l:>4} "
        f"{wr:>7.1%} {profit:>8.2f} {roi:>6.1f}% {conf:>7.1%} {edge or 0.0:>7.1%} {elo:>8.1f}"
    )


def analyze_coverage(rows: list[EnrichedFight]) -> None:
    print("=" * 82)
    print("ELO JOIN COVERAGE")
    print("=" * 82)
    total = len(rows)
    status_counts = Counter(row.join_status for row in rows)
    method_counts = Counter(row.join_method or "none" for row in rows)
    with_elo = len([row for row in rows if row.pick_elo_diff is not None])
    print(f"Rows analyzed:           {total:>5}")
    print(f"Rows with oriented ELO:  {with_elo:>5}  ({with_elo / total:>6.1%})" if total else "Rows with oriented ELO:      0")
    print("\nJoin status")
    for status, count in status_counts.most_common():
        print(f"  {status:<24} {count:>5}")
    print("\nJoin method")
    for method, count in method_counts.most_common():
        print(f"  {method:<24} {count:>5}")
    print()


PICK_ELO_BUCKETS = [
    ("pick -200 or worse", -math.inf, -200),
    ("pick -100 to -199", -200, -100),
    ("pick -50 to -99", -100, -50),
    ("pick -1 to -49", -50, 0),
    ("pick +1 to +49", 0, 50),
    ("pick +50 to +99", 50, 100),
    ("pick +100 to +199", 100, 200),
    ("pick +200 or better", 200, math.inf),
]


def in_range(value: int | None, lo: float, hi: float) -> bool:
    return value is not None and lo <= value < hi


def analyze_pick_diff(rows: list[EnrichedFight]) -> None:
    rows = analysis_rows(rows)
    print("=" * 104)
    print("MODEL PICK PERFORMANCE BY PICK ELO DIFF  (pick_elo - opponent_elo)")
    print("=" * 104)
    print_stats_header()
    for label, lo, hi in PICK_ELO_BUCKETS:
        print_stats_row(label, [row for row in rows if in_range(row.pick_elo_diff, lo, hi)])
    print_stats_row("TOTAL", rows)
    print()


def analyze_agreement(rows: list[EnrichedFight]) -> None:
    rows = analysis_rows(rows)
    print("=" * 104)
    print("MODEL / ELO AGREEMENT")
    print("=" * 104)
    print_stats_header()
    print_stats_row("model pick higher ELO", [row for row in rows if row.model_agrees_with_elo is True])
    print_stats_row("model pick lower ELO", [row for row in rows if row.model_agrees_with_elo is False])
    print_stats_row("ELO tie", [row for row in rows if row.model_agrees_with_elo is None])
    print_stats_row("TOTAL", rows)
    print()


CONFIDENCE_BUCKETS = [
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70%+", 0.70, 1.01),
]

ELO_SUPPORT_BUCKETS = [
    ("ELO against", -math.inf, 0),
    ("ELO +0 to +49", 0, 50),
    ("ELO +50 to +99", 50, 100),
    ("ELO +100+", 100, math.inf),
]


def analyze_low_confidence(rows: list[EnrichedFight]) -> None:
    rows = analysis_rows(rows)
    print("=" * 104)
    print("CONFIDENCE + ELO SUPPORT  (answers: which 51-65% picks deserve upgrades?)")
    print("=" * 104)
    print_stats_header(label_width=20)
    for conf_label, conf_lo, conf_hi in CONFIDENCE_BUCKETS:
        conf_rows = [row for row in rows if conf_lo <= row.pick_prob < conf_hi]
        if not conf_rows:
            continue
        print_stats_row(conf_label, conf_rows, label_width=20)
        for elo_label, elo_lo, elo_hi in ELO_SUPPORT_BUCKETS:
            segment = [row for row in conf_rows if in_range(row.pick_elo_diff, elo_lo, elo_hi)]
            if segment:
                print_stats_row(f"  {elo_label}", segment, label_width=20)
    print()


PURE_ELO_BUCKETS = [
    ("ELO gap 1-49", 1, 50),
    ("ELO gap 50-99", 50, 100),
    ("ELO gap 100-199", 100, 200),
    ("ELO gap 200+", 200, math.inf),
]


def analyze_pure_elo(rows: list[EnrichedFight]) -> None:
    rows = [
        row for row in rows
        if row.join_status == "matched"
        and row.abs_elo_diff is not None
        and row.abs_elo_diff > 0
        and row.elo_pick_correct is not None
    ]
    print("=" * 104)
    print("PURE ELO PICK PERFORMANCE  (bet the higher-ELO fighter at listed odds)")
    print("=" * 104)
    print_stats_header()
    for label, lo, hi in PURE_ELO_BUCKETS:
        print_stats_row(
            label,
            [row for row in rows if in_range(row.abs_elo_diff, lo, hi)],
            pnl_attr="elo_pick_pnl",
            correct_attr="elo_pick_correct",
        )
    print_stats_row("TOTAL", rows, pnl_attr="elo_pick_pnl", correct_attr="elo_pick_correct")
    print()


def analyze_rules(rows: list[EnrichedFight]) -> None:
    rows = analysis_rows(rows)
    print("=" * 104)
    print("CURRENT BET/SKIP RULES BY ELO SUPPORT")
    print("=" * 104)
    print_stats_header()
    print_stats_row("bet + ELO support", [row for row in rows if row.bet and (row.pick_elo_diff or 0) > 0])
    print_stats_row("bet + ELO against", [row for row in rows if row.bet and (row.pick_elo_diff or 0) < 0])
    print_stats_row("skip + ELO +50", [row for row in rows if not row.bet and (row.pick_elo_diff or 0) >= 50])
    print_stats_row("skip + ELO +100", [row for row in rows if not row.bet and (row.pick_elo_diff or 0) >= 100])
    print_stats_row(
        "skip 50-65% + ELO +50",
        [row for row in rows if not row.bet and 0.50 <= row.pick_prob < 0.65 and (row.pick_elo_diff or 0) >= 50],
    )
    print()

    candidates = [
        row for row in rows
        if not row.bet
        and 0.50 <= row.pick_prob < 0.65
        and (row.pick_elo_diff or 0) >= 50
    ]
    if candidates:
        print("Top skipped low-confidence ELO-supported candidates")
        print(f"{'Date':<12} {'Fight':<44} {'Pick':<22} {'Prob':>6} {'Edge':>7} {'ELO+':>6} {'Correct':>7}  Reason")
        print("-" * 122)
        for row in sorted(candidates, key=lambda r: (r.pick_elo_diff or 0, r.pick_prob), reverse=True)[:15]:
            fight = f"{row.fighter1} vs {row.fighter2}"[:44]
            edge = f"{row.edge:+.1%}" if row.edge is not None else "--"
            correct = "W" if row.pick_correct is True else "L" if row.pick_correct is False else "--"
            print(
                f"{row.date:<12} {fight:<44} {row.pick:<22} "
                f"{row.pick_prob:>5.1%} {edge:>7} {row.pick_elo_diff:>6} {correct:>7}  {row.skip_reason}"
            )
        print()


def write_enriched_csv(path: Path, rows: list[EnrichedFight], *, include_unmatched: bool) -> None:
    output_rows = rows if include_unmatched else [row for row in rows if row.join_status == "matched"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(output_rows[0]).keys()) if output_rows else list(EnrichedFight.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in output_rows:
            writer.writerow(asdict(row))
    print(f"Enriched CSV written: {path} ({len(output_rows)} rows)")


def main() -> None:
    args = parse_args()
    aliases = load_aliases(DEFAULT_ALIAS_SOURCES)
    rows = enrich_results(
        args.results,
        args.sidecar,
        aliases,
        date_tolerance_days=args.date_tolerance_days,
    )

    if args.only_bets:
        rows = [row for row in rows if row.bet]

    if not rows:
        print("No usable backtest rows found.")
        return

    if args.write_enriched:
        write_enriched_csv(args.write_enriched, rows, include_unmatched=args.include_unmatched)

    if args.section in {"coverage", "all"}:
        analyze_coverage(rows)
    if args.section in {"pick_diff", "all"}:
        analyze_pick_diff(rows)
    if args.section in {"agreement", "all"}:
        analyze_agreement(rows)
    if args.section in {"low_confidence", "all"}:
        analyze_low_confidence(rows)
    if args.section in {"pure_elo", "all"}:
        analyze_pure_elo(rows)
    if args.section in {"rules", "all"}:
        analyze_rules(rows)


if __name__ == "__main__":
    main()
