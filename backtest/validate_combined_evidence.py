#!/usr/bin/env python3
"""
Validate combined context-evidence families on historical graded rows.

This is still evidence-only. It asks whether combinations like
ELO support + non-expensive price + first-pass trait support are historically
useful, using leave-one-out/temporal pattern evidence where applicable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.context_packet import DEFAULT_POOL, fmt_signed_pct, row_to_dict  # noqa: E402
from backtest.validate_context_pipeline import build_validation_row, summarize  # noqa: E402


RulePredicate = Callable[[dict[str, Any]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--main-db", type=Path, default=ROOT_DIR / "data" / "ufc_database.db")
    parser.add_argument("--mode", choices=("leave-one-out", "temporal", "in-sample"), default="leave-one-out")
    parser.add_argument("--skips-only", action="store_true")
    parser.add_argument("--min-date", default=None, help="Optional YYYY-MM-DD lower bound for target fight dates.")
    parser.add_argument("--max-date", default=None, help="Optional YYYY-MM-DD upper bound for target fight dates.")
    parser.add_argument(
        "--audit-rule",
        default=None,
        choices=[rule[0] for rule in RULES] if "RULES" in globals() else None,
        help="Print artifact breakdowns for one evidence family.",
    )
    parser.add_argument("--show-rows", action="store_true", help="With --audit-rule, print matching rows.")
    parser.add_argument(
        "--dedupe-main-fight",
        action="store_true",
        help="Report rule summaries after keeping one row per main_fight_id.",
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Print a compact table for leave-one-out, temporal, and in-sample modes using the same target window.",
    )
    return parser.parse_args()


def fetch_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = [row_to_dict(row) for row in conn.execute("SELECT * FROM backtest_fight_pool ORDER BY date, id")]
    trait_rows = conn.execute(
        """
        SELECT fight_pool_id, data_json
        FROM evidence_items
        WHERE evidence_type = 'trait_delta'
        """
    ).fetchall()
    traits_by_id = {row["fight_pool_id"]: json.loads(row["data_json"]) for row in trait_rows}
    for row in rows:
        trait = traits_by_id.get(row["id"])
        deltas = {} if trait is None else trait.get("deltas", {})
        row["trait_delta"] = trait
        row["cardio_score_diff"] = deltas.get("cardio_score_diff")
        row["striking_efficiency_score_diff"] = deltas.get("striking_efficiency_score_diff")
        row["defensive_exposure_score_diff"] = deltas.get("defensive_exposure_score_diff")
        row["anti_control_score_diff"] = deltas.get("anti_control_score_diff")
        row["offensive_control_score_diff"] = deltas.get("offensive_control_score_diff")
        row["trait_clean_sample"] = (
            trait is not None
            and (trait.get("trait_confidence") or 0.0) >= 0.6
            and (trait.get("opponent_trait_confidence") or 0.0) >= 0.6
        )
    return rows


def graded_targets(rows: list[dict[str, Any]], *, skips_only: bool) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("pick_correct") is not None and (not skips_only or row.get("bet") is False)
    ]


def row_in_date_window(row: dict[str, Any], *, min_date: str | None, max_date: str | None) -> bool:
    row_date = row.get("date")
    if row_date is None:
        return False
    if min_date is not None and row_date < min_date:
        return False
    if max_date is not None and row_date > max_date:
        return False
    return True


def filtered_targets(
    rows: list[dict[str, Any]],
    *,
    skips_only: bool,
    min_date: str | None,
    max_date: str | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in graded_targets(rows, skips_only=skips_only)
        if row_in_date_window(row, min_date=min_date, max_date=max_date)
    ]


def has_golden_elo(row: dict[str, Any]) -> bool:
    return (
        row.get("bet") is False
        and 0.50 <= (row.get("pick_prob") or 0.0) < 0.65
        and (row.get("pick_elo_diff") or -9999) >= 50
    )


def has_trait_support(row: dict[str, Any]) -> bool:
    return (
        (row.get("cardio_score_diff") is not None and row["cardio_score_diff"] >= 10)
        or (row.get("striking_efficiency_score_diff") is not None and row["striking_efficiency_score_diff"] >= 10)
        or (row.get("defensive_exposure_score_diff") is not None and row["defensive_exposure_score_diff"] <= -10)
    )


def has_cardio_support(row: dict[str, Any]) -> bool:
    return row.get("cardio_score_diff") is not None and row["cardio_score_diff"] >= 10


def has_non_cardio_trait_support(row: dict[str, Any]) -> bool:
    return has_trait_support(row) and not has_cardio_support(row)


def has_trait_caution(row: dict[str, Any]) -> bool:
    return (
        (row.get("cardio_score_diff") is not None and row["cardio_score_diff"] <= -10)
        or (row.get("striking_efficiency_score_diff") is not None and row["striking_efficiency_score_diff"] <= -10)
        or (row.get("defensive_exposure_score_diff") is not None and row["defensive_exposure_score_diff"] >= 10)
    )


RULES: list[tuple[str, str, RulePredicate]] = [
    (
        "golden_elo_skip",
        "Skipped 50-65% model picks with +50 or more ELO support.",
        has_golden_elo,
    ),
    (
        "golden_elo_not_expensive",
        "Golden ELO skip pattern with pick odds better than -300.",
        lambda row: has_golden_elo(row) and row.get("pick_odds") is not None and row["pick_odds"] > -300,
    ),
    (
        "golden_elo_plus_trait_support",
        "Golden ELO skip pattern with cardio/striking-efficiency/lower-exposure trait support.",
        lambda row: has_golden_elo(row) and has_trait_support(row),
    ),
    (
        "golden_elo_not_expensive_plus_trait_support",
        "Golden ELO not-expensive pattern plus trait support.",
        lambda row: has_golden_elo(row)
        and row.get("pick_odds") is not None
        and row["pick_odds"] > -300
        and has_trait_support(row),
    ),
    (
        "golden_elo_not_expensive_plus_cardio",
        "Golden ELO not-expensive pattern plus cardio/late-fight trait support.",
        lambda row: has_golden_elo(row)
        and row.get("pick_odds") is not None
        and row["pick_odds"] > -300
        and has_cardio_support(row),
    ),
    (
        "golden_elo_not_expensive_non_cardio_trait",
        "Golden ELO not-expensive pattern plus non-cardio trait support only.",
        lambda row: has_golden_elo(row)
        and row.get("pick_odds") is not None
        and row["pick_odds"] > -300
        and has_non_cardio_trait_support(row),
    ),
    (
        "golden_elo_with_trait_caution",
        "Golden ELO skip pattern with cardio/striking-efficiency/exposure caution.",
        lambda row: has_golden_elo(row) and has_trait_caution(row),
    ),
    (
        "pattern_score_8_plus_and_trait_support",
        "Pattern score >=8 plus trait support, using selected validation mode for the score.",
        lambda row: row.get("score", 0) >= 8 and has_trait_support(row),
    ),
]


def rule_by_name(rule_name: str) -> tuple[str, str, RulePredicate]:
    for rule in RULES:
        if rule[0] == rule_name:
            return rule
    raise ValueError(f"Unknown rule: {rule_name}")


def grouped(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def odds_bucket(odds: int | None) -> str:
    if odds is None:
        return "unknown"
    if odds <= -300:
        return "<= -300"
    if odds < -200:
        return "-299 to -201"
    if odds < -100:
        return "-200 to -101"
    if odds < 100:
        return "-100 to +99"
    if odds < 200:
        return "+100 to +199"
    return ">= +200"


def confidence_bucket(prob: float | None) -> str:
    if prob is None:
        return "unknown"
    if prob < 0.55:
        return "50-55%"
    if prob < 0.60:
        return "55-60%"
    return "60-65%"


def edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "unknown"
    if edge < -0.10:
        return "< -10%"
    if edge < -0.05:
        return "-10% to -5%"
    if edge < 0:
        return "-5% to 0%"
    if edge < 0.05:
        return "0% to +5%"
    return ">= +5%"


def elo_bucket(diff: int | float | None) -> str:
    if diff is None:
        return "unknown"
    if diff < 100:
        return "+50 to +99"
    if diff < 200:
        return "+100 to +199"
    return "+200 or more"


def trait_support_sources(row: dict[str, Any]) -> str:
    sources = []
    if row.get("cardio_score_diff") is not None and row["cardio_score_diff"] >= 10:
        sources.append("cardio")
    if row.get("striking_efficiency_score_diff") is not None and row["striking_efficiency_score_diff"] >= 10:
        sources.append("striking_eff")
    if row.get("defensive_exposure_score_diff") is not None and row["defensive_exposure_score_diff"] <= -10:
        sources.append("lower_def_exposure")
    return "+".join(sources) if sources else "none"


def main_fight_key(row: dict[str, Any]) -> Any:
    main_fight_id = row.get("main_fight_id")
    key = row["id"] if main_fight_id is None else main_fight_id
    try:
        return int(key)
    except (TypeError, ValueError):
        return str(key)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def source_result_status(row: dict[str, Any]) -> str:
    source_results = row.get("source_results")
    row_num = _as_int(row.get("row_num"))
    if not source_results or row_num is None:
        return "unavailable"
    path = Path(source_results)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return "missing_file"

    with path.open(newline="") as f:
        for line_no, source_row in enumerate(csv.DictReader(f), start=2):
            if line_no != row_num:
                continue
            comparisons = [
                str(source_row.get("date", "")).strip() == str(row.get("date", "")).strip(),
                str(source_row.get("fighter1", "")).strip() == str(row.get("fighter1", "")).strip(),
                str(source_row.get("fighter2", "")).strip() == str(row.get("fighter2", "")).strip(),
                str(source_row.get("pick", "")).strip() == str(row.get("pick", "")).strip(),
                _as_int(source_row.get("pick_odds")) == _as_int(row.get("pick_odds")),
            ]
            source_prob = _as_float(source_row.get("pick_prob"))
            row_prob = _as_float(row.get("pick_prob"))
            if source_prob is not None and row_prob is not None:
                comparisons.append(abs(source_prob - row_prob) < 1e-9)
            return "verified" if all(comparisons) else "mismatch"
    return "missing_line"


def annotate_temporal_segments(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda item: (item["date"], item["id"]))
    total = len(ordered)
    for index, row in enumerate(ordered):
        row["temporal_order"] = index + 1
        if total == 0:
            row["temporal_half"] = "unknown"
            row["temporal_quartile"] = "unknown"
            continue
        half = "first_half" if index < total / 2 else "second_half"
        quartile = min(3, int(index * 4 / total)) + 1
        row["temporal_half"] = half
        row["temporal_quartile"] = f"Q{quartile}"


def dedupe_by_main_fight(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    unique = []
    for row in sorted(rows, key=lambda item: (item["date"], item["id"])):
        fight_key = main_fight_key(row)
        if fight_key in seen:
            continue
        seen.add(fight_key)
        unique.append(row)
    return unique


def duplicate_main_fights(rows: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("main_fight_id") is not None:
            groups[main_fight_key(row)].append(row)
    return {fight_id: group for fight_id, group in groups.items() if len(group) > 1}


def enrich_with_main_db(rows: list[dict[str, Any]], main_db: Path) -> None:
    if not main_db.exists():
        for row in rows:
            row["weight_class"] = "unknown"
            row["bookmakers"] = "unknown"
            row["bookmaker_count"] = 0
            row["db_odds_row_count"] = 0
            row["db_pick_odds_exact_match"] = "no_db_rows"
        return

    conn = sqlite3.connect(main_db)
    conn.row_factory = sqlite3.Row
    try:
        fight_ids = [row["main_fight_id"] for row in rows if row.get("main_fight_id") is not None]
        if not fight_ids:
            return
        placeholders = ", ".join("?" for _ in fight_ids)
        fight_meta = {
            row["id"]: dict(row)
            for row in conn.execute(
                f"""
                SELECT id, weight_class, scheduled_rounds, is_title_fight
                FROM fights
                WHERE id IN ({placeholders})
                """,
                fight_ids,
            )
        }
        bookmaker_rows = conn.execute(
            f"""
            SELECT
                bo.fight_id,
                f1.name AS db_fighter1,
                f2.name AS db_fighter2,
                bo.bookmaker,
                bo.fighter_1_odds,
                bo.fighter_2_odds
            FROM betting_odds bo
            JOIN fights f ON f.id = bo.fight_id
            JOIN fighters f1 ON f1.id = f.fighter_1_id
            JOIN fighters f2 ON f2.id = f.fighter_2_id
            WHERE bo.fight_id IN ({placeholders})
            """,
            fight_ids,
        ).fetchall()
        bookmaker_meta: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for odds_row in bookmaker_rows:
            bookmaker_meta[odds_row["fight_id"]].append(dict(odds_row))
    finally:
        conn.close()

    for row in rows:
        fight_id = row.get("main_fight_id")
        meta = fight_meta.get(fight_id, {})
        odds_rows = bookmaker_meta.get(fight_id, [])
        bookmakers = sorted({odds_row.get("bookmaker") for odds_row in odds_rows if odds_row.get("bookmaker")})
        row["weight_class"] = meta.get("weight_class") or "unknown"
        row["scheduled_rounds"] = meta.get("scheduled_rounds")
        row["is_title_fight"] = bool(meta.get("is_title_fight")) if meta.get("is_title_fight") is not None else False
        row["bookmakers"] = ",".join(bookmakers) if bookmakers else "none"
        row["bookmaker_count"] = len(bookmakers)
        row["db_odds_row_count"] = len(odds_rows)
        pick_odds = _as_int(row.get("pick_odds"))
        if not odds_rows:
            row["db_pick_odds_exact_match"] = "no_db_rows"
            continue
        if pick_odds is None:
            row["db_pick_odds_exact_match"] = "missing_pick_odds"
            continue
        db_pick_odds = []
        for odds_row in odds_rows:
            if str(row.get("pick", "")).strip().lower() == str(odds_row.get("db_fighter1", "")).strip().lower():
                db_pick_odds.append(_as_int(odds_row.get("fighter_1_odds")))
            elif str(row.get("pick", "")).strip().lower() == str(odds_row.get("db_fighter2", "")).strip().lower():
                db_pick_odds.append(_as_int(odds_row.get("fighter_2_odds")))
        row["db_pick_odds_exact_match"] = "exact" if pick_odds in db_pick_odds else "line_diff"


def matching_rows_for_rule(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    skips_only: bool,
    rule_name: str,
    min_date: str | None = None,
    max_date: str | None = None,
) -> list[dict[str, Any]]:
    _, _, predicate = rule_by_name(rule_name)
    targets = filtered_targets(rows, skips_only=skips_only, min_date=min_date, max_date=max_date)
    by_id = {row["id"]: row for row in rows}
    matching = []
    for target in targets:
        validation = build_validation_row(rows, target, mode=mode)
        merged = {**target, **validation, **{key: value for key, value in target.items() if key.endswith("_diff")}}
        merged["trait_delta"] = by_id[target["id"]].get("trait_delta")
        if predicate(merged):
            merged["odds_bucket"] = odds_bucket(merged.get("pick_odds"))
            merged["confidence_bucket"] = confidence_bucket(merged.get("pick_prob"))
            merged["edge_bucket"] = edge_bucket(merged.get("edge"))
            merged["elo_bucket"] = elo_bucket(merged.get("pick_elo_diff"))
            merged["trait_support_sources"] = trait_support_sources(merged)
            merged["source_result_status"] = source_result_status(merged)
            merged["odds_provenance_status"] = (
                "present"
                if merged.get("odds_source_file") or merged.get("bookmaker") or merged.get("odds_timestamp")
                else "legacy_missing"
            )
            matching.append(merged)
    annotate_temporal_segments(matching)
    return matching


def build_rule_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    skips_only: bool,
    dedupe_main_fight: bool = False,
    min_date: str | None = None,
    max_date: str | None = None,
) -> list[dict[str, Any]]:
    targets = filtered_targets(rows, skips_only=skips_only, min_date=min_date, max_date=max_date)
    scored = []
    by_id = {row["id"]: row for row in rows}
    for target in targets:
        validation = build_validation_row(rows, target, mode=mode)
        merged = {**target, **validation, **{key: value for key, value in target.items() if key.endswith("_diff")}}
        merged["trait_delta"] = by_id[target["id"]].get("trait_delta")
        scored.append(merged)

    output = []
    for name, description, predicate in RULES:
        matching = [row for row in scored if predicate(row)]
        stats_rows = dedupe_by_main_fight(matching) if dedupe_main_fight else matching
        stats = summarize(stats_rows)
        output.append(
            {
                "rule": name,
                "description": description,
                **stats,
            }
        )
    return output


def print_report(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    skips_only: bool,
    dedupe_main_fight: bool,
    min_date: str | None,
    max_date: str | None,
) -> None:
    print("=" * 100)
    title_suffix = " (UNIQUE MAIN FIGHTS)" if dedupe_main_fight else ""
    print(f"COMBINED EVIDENCE VALIDATION{title_suffix}")
    print("=" * 100)
    print(f"Mode: {mode}")
    print(f"Scope: {'skips only' if skips_only else 'all graded rows'}")
    if min_date or max_date:
        print(f"Target window: {min_date or 'start'} -> {max_date or 'end'}")
    print(f"Rows: {'deduped by main_fight_id' if dedupe_main_fight else 'raw line rows'}")
    print("| Evidence family | N | W-L | WinRate | PnL | ROI |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        win_rate = "--" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        roi = "--" if row["roi"] is None else fmt_signed_pct(row["roi"])
        print(
            f"| {row['rule']} | {row['n']} | {row['wins']}-{row['losses']} | "
            f"{win_rate} | {row['profit']:+.2f} | {roi} |"
        )


def print_mode_comparison(
    rows: list[dict[str, Any]],
    *,
    skips_only: bool,
    dedupe_main_fight: bool,
    min_date: str | None,
    max_date: str | None,
) -> None:
    print("=" * 100)
    print("COMBINED EVIDENCE MODE COMPARISON")
    print("=" * 100)
    print(f"Scope: {'skips only' if skips_only else 'all graded rows'}")
    if min_date or max_date:
        print(f"Target window: {min_date or 'start'} -> {max_date or 'end'}")
    print("| Mode | Evidence family | N | W-L | WinRate | PnL | ROI |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for mode in ("leave-one-out", "temporal", "in-sample"):
        for row in build_rule_rows(
            rows,
            mode=mode,
            skips_only=skips_only,
            dedupe_main_fight=dedupe_main_fight,
            min_date=min_date,
            max_date=max_date,
        ):
            win_rate = "--" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
            roi = "--" if row["roi"] is None else fmt_signed_pct(row["roi"])
            print(
                f"| {mode} | {row['rule']} | {row['n']} | {row['wins']}-{row['losses']} | "
                f"{win_rate} | {row['profit']:+.2f} | {roi} |"
            )


def print_artifact_group(title: str, rows: list[dict[str, Any]], key: str) -> None:
    print(f"\n{title}")
    print("| Group | N | W-L | WinRate | PnL | ROI |")
    print("|---|---:|---:|---:|---:|---:|")
    for group, group_rows in sorted(grouped(rows, key).items(), key=lambda item: str(item[0])):
        stats = summarize(group_rows)
        win_rate = "--" if stats["win_rate"] is None else f"{stats['win_rate']:.1%}"
        roi = "--" if stats["roi"] is None else fmt_signed_pct(stats["roi"])
        print(
            f"| {group} | {stats['n']} | {stats['wins']}-{stats['losses']} | "
            f"{win_rate} | {stats['profit']:+.2f} | {roi} |"
        )


def print_audit(rows: list[dict[str, Any]], *, rule_name: str, mode: str, show_rows: bool) -> None:
    stats = summarize(rows)
    unique_rows = dedupe_by_main_fight(rows)
    unique_stats = summarize(unique_rows)
    duplicates = duplicate_main_fights(rows)
    print("=" * 100)
    print(f"ARTIFACT AUDIT: {rule_name}")
    print("=" * 100)
    print(f"Mode: {mode}")
    print(
        f"Raw rows: N={stats['n']} W-L={stats['wins']}-{stats['losses']} "
        f"WR={stats['win_rate']:.1%} PnL={stats['profit']:+.2f} ROI={fmt_signed_pct(stats['roi'])}"
        if stats["n"]
        else "Raw rows: N=0"
    )
    print(
        f"Unique main fights: N={unique_stats['n']} W-L={unique_stats['wins']}-{unique_stats['losses']} "
        f"WR={unique_stats['win_rate']:.1%} PnL={unique_stats['profit']:+.2f} ROI={fmt_signed_pct(unique_stats['roi'])}"
        if unique_stats["n"]
        else "Unique main fights: N=0"
    )
    print(f"Duplicate main_fight_id groups: {len(duplicates)}")
    for title, key in [
        ("By source results", "source_results"),
        ("By source result row status", "source_result_status"),
        ("By odds provenance status", "odds_provenance_status"),
        ("By temporal half", "temporal_half"),
        ("By temporal quartile", "temporal_quartile"),
        ("By season", "season"),
        ("By gender flag", "female"),
        ("By odds bucket", "odds_bucket"),
        ("By confidence bucket", "confidence_bucket"),
        ("By edge bucket", "edge_bucket"),
        ("By ELO bucket", "elo_bucket"),
        ("By trait support source", "trait_support_sources"),
        ("By weight class", "weight_class"),
        ("By bookmaker coverage", "bookmaker_count"),
        ("By DB odds rows", "db_odds_row_count"),
        ("By DB exact line match", "db_pick_odds_exact_match"),
    ]:
        print_artifact_group(title, rows, key)

    if duplicates:
        print("\nDuplicate main_fight_id rows")
        print("| main_fight_id | Rows | Fight | Odds/probs |")
        print("|---:|---:|---|---|")
        for fight_id, group_rows in sorted(duplicates.items()):
            fight = f"{group_rows[0]['fighter1']} vs {group_rows[0]['fighter2']}"
            odds = ", ".join(f"id={row['id']} odds={row['pick_odds']} prob={row['pick_prob']:.1%}" for row in group_rows)
            print(f"| {fight_id} | {len(group_rows)} | {fight} | {odds} |")

    if show_rows:
        print("\nRows")
        print("| ID | Src line | Date | Fight | Pick | Odds | Prob | Edge | ELO | Traits | Weight | Books | DB line | Result | PnL |")
        print("|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|---|---|---:|")
        for row in rows:
            result = "W" if row["pick_correct"] else "L"
            source_line = row.get("source_row_key") or f"{row.get('source_results')}:{row.get('row_num')}"
            print(
                f"| {row['id']} | {source_line} | {row['date']} | {row['fighter1']} vs {row['fighter2']} | "
                f"{row['pick']} | {row['pick_odds']} | {row['pick_prob']:.1%} | "
                f"{fmt_signed_pct(row['edge'])} | {row['pick_elo_diff']} | "
                f"{row['trait_support_sources']} | {row.get('weight_class', 'unknown')} | "
                f"{row.get('bookmaker_count', 0)} | {row.get('db_pick_odds_exact_match', 'unknown')} | "
                f"{result} | {row['actual_pnl']:+.2f} |"
            )


def main() -> None:
    args = parse_args()
    if not args.pool.exists():
        raise SystemExit(f"Context pool not found: {args.pool}. Run backtest/build_context_pool.py first.")
    conn = sqlite3.connect(args.pool)
    conn.row_factory = sqlite3.Row
    try:
        rows = fetch_rows(conn)
    finally:
        conn.close()

    report_rows = build_rule_rows(
        rows,
        mode=args.mode,
        skips_only=args.skips_only,
        dedupe_main_fight=args.dedupe_main_fight,
        min_date=args.min_date,
        max_date=args.max_date,
    )
    if args.json_only:
        if args.audit_rule:
            audit_rows = matching_rows_for_rule(
                rows,
                mode=args.mode,
                skips_only=args.skips_only,
                rule_name=args.audit_rule,
                min_date=args.min_date,
                max_date=args.max_date,
            )
            enrich_with_main_db(audit_rows, args.main_db)
            print(json.dumps(audit_rows, indent=2, sort_keys=True))
        else:
            print(json.dumps(report_rows, indent=2, sort_keys=True))
    else:
        if args.compare_modes:
            print_mode_comparison(
                rows,
                skips_only=args.skips_only,
                dedupe_main_fight=args.dedupe_main_fight,
                min_date=args.min_date,
                max_date=args.max_date,
            )
            print()
        print_report(
            report_rows,
            mode=args.mode,
            skips_only=args.skips_only,
            dedupe_main_fight=args.dedupe_main_fight,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        if args.audit_rule:
            audit_rows = matching_rows_for_rule(
                rows,
                mode=args.mode,
                skips_only=args.skips_only,
                rule_name=args.audit_rule,
                min_date=args.min_date,
                max_date=args.max_date,
            )
            enrich_with_main_db(audit_rows, args.main_db)
            print()
            print_audit(audit_rows, rule_name=args.audit_rule, mode=args.mode, show_rows=args.show_rows)


if __name__ == "__main__":
    main()
