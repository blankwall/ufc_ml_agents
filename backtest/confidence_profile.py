from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_FILES = (
    ROOT_DIR / "backtest" / "backtest_2025_results.csv",
    ROOT_DIR / "backtest" / "backtest_2026_results.csv",
)
DEFAULT_SCORE_COUNT = 10
CONFIDENCE_METHOD = "backtest_pick_prob_decile"


@dataclass(frozen=True)
class ConfidenceBand:
    score: int
    min_prob: float
    max_prob: float
    avg_prob: float
    win_rate: float
    sample_size: int


def build_confidence_bands(
    pick_rows: Sequence[tuple[float, bool]],
    score_count: int = DEFAULT_SCORE_COUNT,
) -> list[ConfidenceBand]:
    if score_count <= 0:
        raise ValueError("score_count must be positive")

    ordered = sorted(
        ((float(prob), bool(correct)) for prob, correct in pick_rows),
        key=lambda item: item[0],
    )
    if not ordered:
        raise ValueError("No confidence rows available")

    total = len(ordered)
    bands: list[ConfidenceBand] = []

    for score in range(1, score_count + 1):
        start = (score - 1) * total // score_count
        end = score * total // score_count
        if end <= start:
            continue

        bucket = ordered[start:end]
        probs = [prob for prob, _ in bucket]
        wins = sum(1 for _, correct in bucket if correct)
        bands.append(
            ConfidenceBand(
                score=score,
                min_prob=probs[0],
                max_prob=probs[-1],
                avg_prob=sum(probs) / len(probs),
                win_rate=wins / len(bucket),
                sample_size=len(bucket),
            )
        )

    if not bands:
        raise ValueError("No confidence bands could be built")

    return bands


def load_pick_rows_from_results(result_files: Sequence[Path]) -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []

    for path in result_files:
        if not path.exists():
            raise FileNotFoundError(f"Confidence results file not found: {path}")

        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pick_prob = row.get("pick_prob")
                pick_correct = row.get("pick_correct")
                if not pick_prob or pick_correct not in {"True", "False"}:
                    continue
                rows.append((float(pick_prob), pick_correct == "True"))

    if not rows:
        raise ValueError("No usable pick_prob rows found in confidence results files")

    return rows


@lru_cache(maxsize=1)
def get_default_confidence_bands() -> tuple[ConfidenceBand, ...]:
    return tuple(build_confidence_bands(load_pick_rows_from_results(DEFAULT_RESULTS_FILES)))


def lookup_confidence_band(
    pick_prob: float,
    bands: Sequence[ConfidenceBand] | None = None,
) -> ConfidenceBand:
    if bands is None:
        bands = get_default_confidence_bands()
    if not bands:
        raise ValueError("No confidence bands available")

    normalized_prob = float(pick_prob)
    if normalized_prob <= bands[0].max_prob:
        return bands[0]
    if normalized_prob >= bands[-1].min_prob:
        return bands[-1]

    for band in bands:
        if band.min_prob <= normalized_prob <= band.max_prob:
            return band
        if normalized_prob < band.max_prob:
            return band

    return bands[-1]


def describe_confidence(
    pick_prob: float,
    bands: Sequence[ConfidenceBand] | None = None,
) -> dict[str, int | float | str]:
    band = lookup_confidence_band(pick_prob, bands=bands)
    return {
        "confidence_score": band.score,
        "confidence_method": CONFIDENCE_METHOD,
        "confidence_prob_min": round(band.min_prob * 100, 1),
        "confidence_prob_max": round(band.max_prob * 100, 1),
        "confidence_avg_prob": round(band.avg_prob * 100, 1),
        "confidence_historical_win_rate": round(band.win_rate * 100, 1),
        "confidence_sample_size": band.sample_size,
    }
