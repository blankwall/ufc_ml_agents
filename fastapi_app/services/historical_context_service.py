from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from backtest.confidence_profile import DEFAULT_RESULTS_FILES

MIN_PRIMARY_SAMPLE = 8


@dataclass(frozen=True)
class HistoricalPick:
    correct: bool
    pnl: float
    pick_odds: float
    pick_prob: float
    edge: float
    female: bool

    @property
    def side(self) -> str:
        return "favorite" if self.pick_odds < 0 else "underdog"

    @property
    def odds_bucket(self) -> str:
        return _odds_bucket(self.pick_odds)

    @property
    def model_prob_bucket(self) -> str:
        return _model_prob_bucket(self.pick_prob)

    @property
    def edge_bucket(self) -> str:
        return _edge_bucket(self.edge)


def _market_implied_prob(american_odds: float) -> float:
    if american_odds >= 100:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _odds_bucket(american_odds: float) -> str:
    if american_odds < -400:
        return "favorite <-400"
    if american_odds < -300:
        return "favorite -300 to -400"
    if american_odds < -200:
        return "favorite -200 to -300"
    if american_odds < 0:
        return "favorite -100 to -200"
    if american_odds < 200:
        return "underdog +100 to +200"
    if american_odds < 300:
        return "underdog +200 to +300"
    return "underdog +300+"


def _model_prob_bucket(pick_prob: float) -> str:
    pct = pick_prob * 100
    if pct < 55:
        return "model prob 50-55%"
    if pct < 60:
        return "model prob 55-60%"
    if pct < 65:
        return "model prob 60-65%"
    if pct < 70:
        return "model prob 65-70%"
    if pct < 75:
        return "model prob 70-75%"
    if pct < 80:
        return "model prob 75-80%"
    return "model prob 80%+"


def _edge_bucket(edge: float) -> str:
    pct = edge * 100
    if pct < 0:
        return "edge <0%"
    if pct < 5:
        return "edge 0-5%"
    if pct < 10:
        return "edge 5-10%"
    if pct < 15:
        return "edge 10-15%"
    if pct < 20:
        return "edge 15-20%"
    return "edge 20%+"


def _read_result_rows(path: Path) -> list[HistoricalPick]:
    rows: list[HistoricalPick] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("pick_correct") not in {"True", "False"}:
                continue
            pick_prob = row.get("pick_prob")
            pick_odds = row.get("pick_odds")
            if not pick_prob or not pick_odds:
                continue

            odds = float(pick_odds)
            prob = float(pick_prob)
            rows.append(
                HistoricalPick(
                    correct=row["pick_correct"] == "True",
                    pnl=float(row["actual_pnl"]) if row.get("actual_pnl") else 0.0,
                    pick_odds=odds,
                    pick_prob=prob,
                    edge=prob - _market_implied_prob(odds),
                    female=row.get("female") == "True",
                )
            )
    return rows


@lru_cache(maxsize=1)
def get_default_historical_picks() -> tuple[HistoricalPick, ...]:
    rows: list[HistoricalPick] = []
    for path in DEFAULT_RESULTS_FILES:
        rows.extend(_read_result_rows(path))
    if not rows:
        raise ValueError("No usable historical pick rows found")
    return tuple(rows)


def _stats(label: str, rows: list[HistoricalPick], *, source: str, criteria: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    wins = sum(1 for row in rows if row.correct)
    profit = sum(row.pnl for row in rows)
    return {
        "label": label,
        "source": source,
        "criteria": criteria,
        "sample_size": count,
        "wins": wins,
        "losses": count - wins,
        "win_rate": round((wins / count) * 100, 1) if count else None,
        "roi": round((profit / count) * 100, 1) if count else None,
        "profit": round(profit, 3),
        "avg_model_prob": round((sum(row.pick_prob for row in rows) / count) * 100, 1) if count else None,
        "avg_edge": round((sum(row.edge for row in rows) / count) * 100, 1) if count else None,
        "avg_pick_odds": round(sum(row.pick_odds for row in rows) / count, 1) if count else None,
    }


def _match_stats(
    *,
    label: str,
    rows: tuple[HistoricalPick, ...],
    criteria: list[dict[str, Any]],
    predicates: list[Callable[[HistoricalPick], bool]],
) -> dict[str, Any]:
    matched = [row for row in rows if all(predicate(row) for predicate in predicates)]
    return _stats(label, matched, source="backtest_2025_2026", criteria=criteria)


def _criteria(field: str, value: Any) -> dict[str, Any]:
    return {"field": field, "value": value}


def describe_historical_context(
    *,
    pick_model_prob: float,
    pick_market_prob: float,
    pick_odds: int | None,
    is_wmma: bool | None,
    rows: tuple[HistoricalPick, ...] | None = None,
) -> dict[str, Any]:
    historical_rows = rows if rows is not None else get_default_historical_picks()
    edge = pick_model_prob - pick_market_prob
    side = None if pick_odds is None else ("favorite" if pick_odds < 0 else "underdog")
    odds_bucket = _odds_bucket(float(pick_odds)) if pick_odds is not None else None
    model_bucket = _model_prob_bucket(pick_model_prob)
    edge_bucket = _edge_bucket(edge)

    primary_candidates: list[tuple[str, list[dict[str, Any]], list[Callable[[HistoricalPick], bool]]]] = []
    if odds_bucket is not None and side is not None:
        primary_candidates.extend(
            [
                (
                    f"{odds_bucket}, {model_bucket}, {edge_bucket}",
                    [
                        _criteria("side", side),
                        _criteria("odds_bucket", odds_bucket),
                        _criteria("model_prob_bucket", model_bucket),
                        _criteria("edge_bucket", edge_bucket),
                    ],
                    [
                        lambda row, side=side: row.side == side,
                        lambda row, odds_bucket=odds_bucket: row.odds_bucket == odds_bucket,
                        lambda row, model_bucket=model_bucket: row.model_prob_bucket == model_bucket,
                        lambda row, edge_bucket=edge_bucket: row.edge_bucket == edge_bucket,
                    ],
                ),
                (
                    f"{odds_bucket}, {edge_bucket}",
                    [
                        _criteria("side", side),
                        _criteria("odds_bucket", odds_bucket),
                        _criteria("edge_bucket", edge_bucket),
                    ],
                    [
                        lambda row, side=side: row.side == side,
                        lambda row, odds_bucket=odds_bucket: row.odds_bucket == odds_bucket,
                        lambda row, edge_bucket=edge_bucket: row.edge_bucket == edge_bucket,
                    ],
                ),
                (
                    f"{odds_bucket}, {model_bucket}",
                    [
                        _criteria("side", side),
                        _criteria("odds_bucket", odds_bucket),
                        _criteria("model_prob_bucket", model_bucket),
                    ],
                    [
                        lambda row, side=side: row.side == side,
                        lambda row, odds_bucket=odds_bucket: row.odds_bucket == odds_bucket,
                        lambda row, model_bucket=model_bucket: row.model_prob_bucket == model_bucket,
                    ],
                ),
                (
                    odds_bucket,
                    [_criteria("side", side), _criteria("odds_bucket", odds_bucket)],
                    [
                        lambda row, side=side: row.side == side,
                        lambda row, odds_bucket=odds_bucket: row.odds_bucket == odds_bucket,
                    ],
                ),
            ]
        )

    primary_candidates.append(
        (
            f"{side.title()}, {model_bucket}, {edge_bucket}" if side is not None else f"{model_bucket}, {edge_bucket}",
            (
                [_criteria("side", side), _criteria("model_prob_bucket", model_bucket), _criteria("edge_bucket", edge_bucket)]
                if side is not None
                else [_criteria("model_prob_bucket", model_bucket), _criteria("edge_bucket", edge_bucket)]
            ),
            (
                [
                    lambda row, side=side: row.side == side,
                    lambda row, model_bucket=model_bucket: row.model_prob_bucket == model_bucket,
                    lambda row, edge_bucket=edge_bucket: row.edge_bucket == edge_bucket,
                ]
                if side is not None
                else [
                    lambda row, model_bucket=model_bucket: row.model_prob_bucket == model_bucket,
                    lambda row, edge_bucket=edge_bucket: row.edge_bucket == edge_bucket,
                ]
            ),
        )
    )
    if is_wmma is True:
        primary_candidates.insert(
            0,
            (
                f"WMMA, {model_bucket}, {edge_bucket}",
                [_criteria("is_wmma", True), _criteria("model_prob_bucket", model_bucket), _criteria("edge_bucket", edge_bucket)],
                [
                    lambda row: row.female is True,
                    lambda row, model_bucket=model_bucket: row.model_prob_bucket == model_bucket,
                    lambda row, edge_bucket=edge_bucket: row.edge_bucket == edge_bucket,
                ],
            ),
        )

    primary_options = [
        _match_stats(label=label, rows=historical_rows, criteria=criteria, predicates=predicates)
        for label, criteria, predicates in primary_candidates
    ]
    primary_bucket = next((item for item in primary_options if item["sample_size"] >= MIN_PRIMARY_SAMPLE), primary_options[0])

    return {
        "primary_bucket": primary_bucket,
    }
