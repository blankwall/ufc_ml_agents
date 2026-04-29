from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.confidence_profile import build_confidence_bands, describe_confidence  # noqa: E402


def test_build_confidence_bands_splits_into_monotonic_scores():
    rows = [
        (0.51, False),
        (0.52, True),
        (0.53, False),
        (0.54, True),
        (0.61, True),
        (0.62, True),
        (0.63, False),
        (0.64, True),
        (0.74, True),
        (0.75, True),
    ]

    bands = build_confidence_bands(rows, score_count=5)

    assert [band.score for band in bands] == [1, 2, 3, 4, 5]
    assert all(left.max_prob <= right.min_prob for left, right in zip(bands, bands[1:]))
    assert sum(band.sample_size for band in bands) == len(rows)


def test_describe_confidence_uses_matching_band():
    bands = build_confidence_bands(
        [
            (0.51, False),
            (0.52, True),
            (0.61, True),
            (0.62, True),
            (0.74, True),
            (0.75, True),
        ],
        score_count=3,
    )

    low = describe_confidence(0.515, bands=bands)
    mid = describe_confidence(0.615, bands=bands)
    high = describe_confidence(0.90, bands=bands)

    assert low["confidence_score"] == 1
    assert mid["confidence_score"] == 2
    assert high["confidence_score"] == 3
    assert high["confidence_sample_size"] == 2
