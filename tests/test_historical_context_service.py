from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from services.historical_context_service import HistoricalPick, describe_historical_context  # noqa: E402


def test_historical_context_returns_primary_bucket_only():
    rows = tuple(
        HistoricalPick(
            correct=i < 5,
            pnl=1.5 if i < 5 else -1.0,
            pick_odds=150,
            pick_prob=0.57,
            edge=0.17,
            female=False,
        )
        for i in range(8)
    )

    context = describe_historical_context(
        pick_model_prob=0.567,
        pick_market_prob=0.38,
        pick_odds=150,
        is_wmma=False,
        rows=rows,
    )

    assert context["primary_bucket"]["label"] == "underdog +100 to +200, model prob 55-60%, edge 15-20%"
    assert context["primary_bucket"]["sample_size"] == 8
    assert context["primary_bucket"]["wins"] == 5
    assert context["primary_bucket"]["losses"] == 3
    assert context["primary_bucket"]["win_rate"] == 62.5
    assert context["primary_bucket"]["roi"] == 56.2
    assert set(context) == {"primary_bucket"}


def test_historical_context_does_not_include_supporting_buckets():
    rows = tuple(
        HistoricalPick(
            correct=True,
            pnl=0.5,
            pick_odds=-150,
            pick_prob=0.62,
            edge=0.02,
            female=False,
        )
        for _ in range(8)
    )

    context = describe_historical_context(
        pick_model_prob=0.62,
        pick_market_prob=0.60,
        pick_odds=-150,
        is_wmma=False,
        rows=rows,
    )

    assert set(context) == {"primary_bucket"}
    assert context["primary_bucket"]["label"] == "favorite -100 to -200, model prob 60-65%, edge 0-5%"
