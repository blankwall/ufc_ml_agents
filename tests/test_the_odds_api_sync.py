import csv
import json
import sys
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fastapi_app"))

from fastapi_app.routers import events as events_router  # noqa: E402
from fastapi_app.services import predict_service  # noqa: E402
from fastapi_app.services import sherdog_recovery_service  # noqa: E402
from fastapi_app.services import the_odds_api_service as odds_service  # noqa: E402


def _fixed_now():
    return datetime(2026, 5, 18, 12, 0, 0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_the_odds_api_sync_adds_only_unknown_fights(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    raw_dir = tmp_path / "data" / "raw" / "the_odds_api"
    store_path = odds_dir / "the_odds_api_events.json"
    output_csv = odds_dir / "the_odds_api_new_events.csv"
    state_path = odds_dir / "the_odds_api_sync.json"

    _write_csv(
        odds_dir / "ufc-existing.csv",
        [
            {
                "event_name": "UFC Existing",
                "event_date": "May 1st",
                "event_url": "https://example.com/ufc-existing",
                "fighter1": "Alex Pereira",
                "fighter2": "Max Holloway",
                "fighter1_odds": -150,
                "fighter2_odds": 130,
                "fighter1_prob": 0.6,
                "fighter2_prob": 0.4348,
            }
        ],
    )

    monkeypatch.setattr(odds_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(odds_service, "USER_EVENTS_DIR", user_dir)
    monkeypatch.setattr(odds_service, "RAW_DIR", raw_dir)
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "OUTPUT_CSV", output_csv)
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)
    monkeypatch.setattr(odds_service, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(odds_service, "recovery_enabled", lambda: True)
    monkeypatch.setattr(
        odds_service,
        "recover_missing_fighters_from_odds",
        lambda **_kwargs: {"recovered": 1, "queued": 1, "trigger": "the_odds_api_sync"},
    )

    payload = [
        {
            "id": "evt-known",
            "commence_time": "2026-06-01T22:00:00Z",
            "home_team": "Alex Pereira",
            "away_team": "Max Holloway",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:00:00Z",
                            "outcomes": [
                                {"name": "Alex Pereira", "price": -180},
                                {"name": "Max Holloway", "price": 150},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "evt-new",
            "commence_time": "2026-06-08T22:00:00Z",
            "home_team": "Carlos Prates",
            "away_team": "Michael Morales",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:05:00Z",
                            "outcomes": [
                                {"name": "Carlos Prates", "price": 145},
                                {"name": "Michael Morales", "price": -170},
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload, {"x-requests-remaining": "42", "x-requests-used": "8", "x-requests-last": "1"}),
    )

    result = odds_service.sync_new_the_odds_api_events(api_key="test-key")

    assert result["new_rows_added"] == 1
    assert result["skipped_existing"] == 1
    assert store_path.exists()
    assert output_csv.exists()
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert rows[0]["fighter1"] == "Carlos Prates"
    assert rows[0]["fighter2"] == "Michael Morales"
    assert rows[0]["bookmaker"] == "FanDuel"

    store = json.loads(store_path.read_text())
    assert len(store["events"]) == 1
    fight = store["events"][0]["fights"][0]
    assert fight["event_name"] == "MMA Card · 2026-06-08"
    assert fight["event_date"] == "2026-06-08"
    assert fight["active"] is True
    assert len(fight["odds_history"]) == 1
    assert fight["odds_history"][0]["fighter1_odds"] == 145

    state = json.loads(state_path.read_text())
    assert state["new_rows_added"] == 1
    assert state["sherdog_recovery"]["recovered"] == 1
    assert Path(tmp_path / state["raw_snapshot"]).exists()
    assert state["store_json"] == "data/future_fight_odds/the_odds_api_events.json"


def test_the_odds_api_sync_updates_existing_fight_history(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    raw_dir = tmp_path / "data" / "raw" / "the_odds_api"
    store_path = odds_dir / "the_odds_api_events.json"
    output_csv = odds_dir / "the_odds_api_new_events.csv"
    state_path = odds_dir / "the_odds_api_sync.json"

    monkeypatch.setattr(odds_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(odds_service, "USER_EVENTS_DIR", user_dir)
    monkeypatch.setattr(odds_service, "RAW_DIR", raw_dir)
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "OUTPUT_CSV", output_csv)
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)
    monkeypatch.setattr(odds_service, "ROOT_DIR", tmp_path)

    payload_one = [
        {
            "id": "evt-new",
            "commence_time": "2026-06-08T22:00:00Z",
            "home_team": "Carlos Prates",
            "away_team": "Michael Morales",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:05:00Z",
                            "outcomes": [
                                {"name": "Carlos Prates", "price": 145},
                                {"name": "Michael Morales", "price": -170},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    payload_two = [
        {
            "id": "evt-new",
            "commence_time": "2026-06-08T22:00:00Z",
            "home_team": "Carlos Prates",
            "away_team": "Michael Morales",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-18T12:05:00Z",
                            "outcomes": [
                                {"name": "Carlos Prates", "price": 125},
                                {"name": "Michael Morales", "price": -145},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload_one, {"x-requests-remaining": "42", "x-requests-used": "8", "x-requests-last": "1"}),
    )
    first = odds_service.sync_new_the_odds_api_events(api_key="test-key")
    assert first["new_rows_added"] == 1

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload_two, {"x-requests-remaining": "41", "x-requests-used": "9", "x-requests-last": "1"}),
    )
    second = odds_service.sync_new_the_odds_api_events(api_key="test-key")

    assert second["new_rows_added"] == 0
    assert second["updated_rows"] == 1
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert rows[0]["fighter1_odds"] == "125"
    assert rows[0]["fighter2_odds"] == "-145"

    store = json.loads(store_path.read_text())
    fight = store["events"][0]["fights"][0]
    assert fight["fighter1_odds"] == 125
    assert fight["fighter2_odds"] == -145
    assert len(fight["odds_history"]) == 2
    assert fight["odds_history"][0]["fighter1_odds"] == 145
    assert fight["odds_history"][1]["fighter1_odds"] == 125


def test_the_odds_api_sync_deactivates_removed_fights(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    raw_dir = tmp_path / "data" / "raw" / "the_odds_api"
    store_path = odds_dir / "the_odds_api_events.json"
    output_csv = odds_dir / "the_odds_api_new_events.csv"
    state_path = odds_dir / "the_odds_api_sync.json"

    monkeypatch.setattr(odds_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(odds_service, "USER_EVENTS_DIR", user_dir)
    monkeypatch.setattr(odds_service, "RAW_DIR", raw_dir)
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "OUTPUT_CSV", output_csv)
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)
    monkeypatch.setattr(odds_service, "ROOT_DIR", tmp_path)

    payload_one = [
        {
            "id": "evt-a",
            "commence_time": "2026-06-08T22:00:00Z",
            "home_team": "Carlos Prates",
            "away_team": "Michael Morales",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:05:00Z",
                            "outcomes": [
                                {"name": "Carlos Prates", "price": 145},
                                {"name": "Michael Morales", "price": -170},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "evt-b",
            "commence_time": "2026-06-08T22:00:00Z",
            "home_team": "Song Yadong",
            "away_team": "Deiveson Figueiredo",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:15:00Z",
                            "outcomes": [
                                {"name": "Song Yadong", "price": -125},
                                {"name": "Deiveson Figueiredo", "price": 105},
                            ],
                        }
                    ],
                }
            ],
        },
    ]
    payload_two = [payload_one[0]]

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload_one, {"x-requests-remaining": "42", "x-requests-used": "8", "x-requests-last": "1"}),
    )
    odds_service.sync_new_the_odds_api_events(api_key="test-key")

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload_two, {"x-requests-remaining": "41", "x-requests-used": "9", "x-requests-last": "1"}),
    )
    result = odds_service.sync_new_the_odds_api_events(api_key="test-key")

    assert result["deactivated_rows"] == 1
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert rows[0]["fighter1"] == "Carlos Prates"

    store = json.loads(store_path.read_text())
    fights = {fight["fight_key"]: fight for fight in store["events"][0]["fights"]}
    removed = fights[odds_service.fight_key("Song Yadong", "Deiveson Figueiredo")]
    assert removed["active"] is False
    assert removed["removed_at"] is not None


def test_fetch_odds_payload_uses_default_window_params(monkeypatch):
    captured = {}

    class _Resp:
        headers = {
            "x-requests-remaining": "100",
            "x-requests-used": "1",
            "x-requests-last": "1",
        }

        def raise_for_status(self):
            return None

        def json(self):
            return []

    def _fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(odds_service.requests, "get", _fake_get)
    monkeypatch.setenv(odds_service.WINDOW_DAYS_ENV, "31")

    payload, headers = odds_service.fetch_odds_payload("abc123")

    assert payload == []
    assert headers["x-requests-remaining"] == "100"
    assert captured["url"] == odds_service.ODDS_API_URL
    assert captured["params"]["apiKey"] == "abc123"
    assert captured["params"]["commenceTimeFrom"].endswith("Z")
    assert captured["params"]["commenceTimeTo"].endswith("Z")
    assert captured["timeout"] == 30


def test_the_odds_api_sync_skips_fights_outside_default_window(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    raw_dir = tmp_path / "data" / "raw" / "the_odds_api"
    store_path = odds_dir / "the_odds_api_events.json"
    output_csv = odds_dir / "the_odds_api_new_events.csv"
    state_path = odds_dir / "the_odds_api_sync.json"

    monkeypatch.setattr(odds_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(odds_service, "USER_EVENTS_DIR", user_dir)
    monkeypatch.setattr(odds_service, "RAW_DIR", raw_dir)
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "OUTPUT_CSV", output_csv)
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)
    monkeypatch.setattr(odds_service, "ROOT_DIR", tmp_path)
    monkeypatch.setenv(odds_service.WINDOW_DAYS_ENV, "31")

    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(odds_service, "_utc_now", lambda: now)

    payload = [
        {
            "id": "near-term",
            "commence_time": "2026-06-10T22:00:00Z",
            "home_team": "Carlos Prates",
            "away_team": "Michael Morales",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:05:00Z",
                            "outcomes": [
                                {"name": "Carlos Prates", "price": 145},
                                {"name": "Michael Morales", "price": -170},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "far-out",
            "commence_time": "2026-08-25T22:00:00Z",
            "home_team": "Song Yadong",
            "away_team": "Deiveson Figueiredo",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-05-17T12:15:00Z",
                            "outcomes": [
                                {"name": "Song Yadong", "price": -125},
                                {"name": "Deiveson Figueiredo", "price": 105},
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    monkeypatch.setattr(
        odds_service,
        "fetch_odds_payload",
        lambda _api_key: (payload, {"x-requests-remaining": "42", "x-requests-used": "8", "x-requests-last": "1"}),
    )

    result = odds_service.sync_new_the_odds_api_events(api_key="test-key")

    assert result["new_rows_added"] == 1
    assert result["skipped_invalid"] == 1
    assert result["window_days"] == 31
    rows = list(csv.DictReader(output_csv.open()))
    assert len(rows) == 1
    assert rows[0]["fighter1"] == "Carlos Prates"


def test_bootstrap_store_from_existing_output_csv(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    store_path = odds_dir / "the_odds_api_events.json"
    output_csv = odds_dir / "the_odds_api_new_events.csv"
    state_path = odds_dir / "the_odds_api_sync.json"

    _write_csv(
        output_csv,
        [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "event_url": "",
                "fighter1": "Song Yadong",
                "fighter2": "Deiveson Figueiredo",
                "fighter1_odds": -125,
                "fighter2_odds": 105,
                "fighter1_prob": 0.5556,
                "fighter2_prob": 0.4878,
                "source_event_id": "evt-b",
                "bookmaker": "FanDuel",
                "last_update": "2026-05-17T12:15:00Z",
                "commence_time": "2026-05-30T11:00:00Z",
            }
        ],
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_success_at": "2026-05-17T13:46:10Z"}))

    monkeypatch.setattr(odds_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "OUTPUT_CSV", output_csv)
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)

    store = odds_service._load_store()

    assert len(store["events"]) == 1
    fight = store["events"][0]["fights"][0]
    assert fight["fighter1"] == "Song Yadong"
    assert fight["event_name"] == "MMA Card · 2026-05-30"
    assert len(fight["odds_history"]) == 1
    assert fight["odds_history"][0]["fighter1_odds"] == -125


def test_predict_service_loads_the_odds_api_csv_without_overriding_manual_csv(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    user_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        odds_dir / "ufc-manual.csv",
        [
            {
                "event_name": "UFC Manual",
                "event_date": "May 1st",
                "event_url": "https://example.com/manual",
                "fighter1": "Sean O'Malley",
                "fighter2": "Song Yadong",
                "fighter1_odds": -225,
                "fighter2_odds": 185,
                "fighter1_prob": 0.6923,
                "fighter2_prob": 0.3509,
            }
        ],
    )
    _write_csv(
        odds_dir / "the_odds_api_new_events.csv",
        [
            {
                "event_name": "MMA Card · 2026-06-08 22:00 UTC",
                "event_date": "2026-06-08 22:00:00",
                "event_url": "",
                "fighter1": "Sean O'Malley",
                "fighter2": "Song Yadong",
                "fighter1_odds": -180,
                "fighter2_odds": 150,
                "fighter1_prob": 0.6429,
                "fighter2_prob": 0.4,
                "source_event_id": "evt-1",
                "bookmaker": "FanDuel",
                "last_update": "2026-05-17T12:00:00Z",
                "commence_time": "2026-06-08T22:00:00Z",
            },
            {
                "event_name": "MMA Card · 2026-06-08 22:00 UTC",
                "event_date": "2026-06-08 22:00:00",
                "event_url": "",
                "fighter1": "Carlos Prates",
                "fighter2": "Michael Morales",
                "fighter1_odds": 145,
                "fighter2_odds": -170,
                "fighter1_prob": 0.4082,
                "fighter2_prob": 0.6296,
                "source_event_id": "evt-2",
                "bookmaker": "FanDuel",
                "last_update": "2026-05-17T12:05:00Z",
                "commence_time": "2026-06-08T22:00:00Z",
            },
        ],
    )

    monkeypatch.setattr(predict_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(predict_service, "USER_EVENTS_DIR", user_dir)

    df = predict_service._load_all_odds()

    assert len(df) == 2
    manual = df[(df["fighter1"] == "Sean O'Malley") & (df["fighter2"] == "Song Yadong")].iloc[0]
    added = df[(df["fighter1"] == "Carlos Prates") & (df["fighter2"] == "Michael Morales")].iloc[0]
    assert int(manual["fighter1_odds"]) == -225
    assert manual["source_type"] == "csv"
    assert added["source_type"] == "the_odds_api"


def test_predict_service_user_added_still_overrides_the_odds_api(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    user_dir = tmp_path / "data" / "user_events"
    user_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        odds_dir / "the_odds_api_new_events.csv",
        [
            {
                "event_name": "MMA Card · 2026-06-08 22:00 UTC",
                "event_date": "2026-06-08 22:00:00",
                "event_url": "",
                "fighter1": "Carlos Prates",
                "fighter2": "Michael Morales",
                "fighter1_odds": 145,
                "fighter2_odds": -170,
                "fighter1_prob": 0.4082,
                "fighter2_prob": 0.6296,
                "source_event_id": "evt-2",
                "bookmaker": "FanDuel",
                "last_update": "2026-05-17T12:05:00Z",
                "commence_time": "2026-06-08T22:00:00Z",
            },
        ],
    )
    (user_dir / "manual.json").write_text(
        json.dumps(
            {
                "fights": [
                    {
                        "event_name": "UFC User Added",
                        "event_date": "June 8th",
                        "event_url": "https://example.com/manual",
                        "fighter1": "Carlos Prates",
                        "fighter2": "Michael Morales",
                        "fighter1_odds": 125,
                        "fighter2_odds": -145,
                        "fighter1_prob": 0.4444,
                        "fighter2_prob": 0.5918,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(predict_service, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(predict_service, "USER_EVENTS_DIR", user_dir)

    df = predict_service._load_all_odds()
    row = df[(df["fighter1"] == "Carlos Prates") & (df["fighter2"] == "Michael Morales")].iloc[0]

    assert int(row["fighter1_odds"]) == 125
    assert row["source_type"] == "user_added"


def test_the_odds_api_rows_group_by_calendar_date(monkeypatch):
    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-05-30 08:00 UTC",
                "event_date": "2026-05-30 08:00:00",
                "event_url": float("nan"),
                "fighter1": "Loma Lookboonmee",
                "fighter2": "Jaqueline Amorim",
                "fighter1_odds": -150,
                "fighter2_odds": 130,
                "fighter1_prob": 0.6,
                "fighter2_prob": 0.4348,
                "source_type": "the_odds_api",
                "source_file": "the_odds_api_new_events.csv",
            },
            {
                "event_name": "MMA Card · 2026-05-30 11:00 UTC",
                "event_date": "2026-05-30 11:00:00",
                "event_url": float("nan"),
                "fighter1": "Song Yadong",
                "fighter2": "Deiveson Figueiredo",
                "fighter1_odds": -125,
                "fighter2_odds": 105,
                "fighter1_prob": 0.5556,
                "fighter2_prob": 0.4878,
                "source_type": "the_odds_api",
                "source_file": "the_odds_api_new_events.csv",
            },
        ]
    )

    monkeypatch.setattr(predict_service, "_resolve_fighter", lambda *_args, **_kwargs: None)

    events_map, _ = predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=predict_service.pd.DataFrame(),
        cache={},
        session=object(),
        extractor=object(),
    )

    assert len(events_map) == 1
    assert "the_odds_api|2026-05-30" in events_map
    grouped = events_map["the_odds_api|2026-05-30"]
    assert grouped["event_name"] == "MMA Card · 2026-05-30"
    assert grouped["event_date"] == "2026-05-30"
    assert len(grouped["fights"]) == 2


def test_scheduler_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv(odds_service.API_KEY_ENV, raising=False)
    monkeypatch.delenv(odds_service.AUTO_SYNC_ENV, raising=False)

    assert odds_service.scheduler_enabled() is False


def test_sync_due_respects_recent_success(monkeypatch, tmp_path):
    state_path = tmp_path / "the_odds_api_sync.json"
    monkeypatch.setattr(odds_service, "STATE_PATH", state_path)
    monkeypatch.setenv(odds_service.SYNC_INTERVAL_HOURS_ENV, "24")

    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    state_path.write_text(json.dumps({"last_success_at": recent.isoformat().replace("+00:00", "Z")}))

    assert odds_service.sync_due(now=datetime.now(timezone.utc)) is False


def test_sync_if_due_noops_when_disabled(monkeypatch):
    monkeypatch.delenv(odds_service.API_KEY_ENV, raising=False)
    monkeypatch.setattr(odds_service, "sync_new_the_odds_api_events", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not sync")))

    assert odds_service.sync_if_due() is None


def test_get_sampled_odds_history_returns_first_middle_latest(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    store_path = odds_dir / "the_odds_api_events.json"
    odds_dir.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_key": "the_odds_api|2026-05-30",
                        "event_name": "MMA Card · 2026-05-30",
                        "event_date": "2026-05-30",
                        "fights": [
                            {
                                "fight_key": odds_service.fight_key("Song Yadong", "Deiveson Figueiredo"),
                                "fighter1": "Song Yadong",
                                "fighter2": "Deiveson Figueiredo",
                                "odds_history": [
                                    {
                                        "captured_at": "2026-05-10T12:00:00Z",
                                        "fighter1_odds": -125,
                                        "fighter2_odds": 105,
                                        "fighter1_prob": 0.5556,
                                        "fighter2_prob": 0.4878,
                                        "bookmaker": "FanDuel",
                                        "last_update": "2026-05-10T12:00:00Z",
                                    },
                                    {
                                        "captured_at": "2026-05-15T12:00:00Z",
                                        "fighter1_odds": -135,
                                        "fighter2_odds": 114,
                                        "fighter1_prob": 0.5745,
                                        "fighter2_prob": 0.4673,
                                        "bookmaker": "FanDuel",
                                        "last_update": "2026-05-15T12:00:00Z",
                                    },
                                    {
                                        "captured_at": "2026-05-20T12:00:00Z",
                                        "fighter1_odds": -145,
                                        "fighter2_odds": 122,
                                        "fighter1_prob": 0.5918,
                                        "fighter2_prob": 0.4505,
                                        "bookmaker": "FanDuel",
                                        "last_update": "2026-05-20T12:00:00Z",
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)

    result = odds_service.get_sampled_odds_history(
        event_date="2026-05-30",
        fighter1="Song Yadong",
        fighter2="Deiveson Figueiredo",
    )

    assert result is not None
    assert result["event_name"] == "MMA Card · 2026-05-30"
    assert result["history_count"] == 3
    assert result["real_history_count"] == 3
    assert result["uses_estimated_samples"] is False
    assert [sample["label"] for sample in result["samples"]] == ["First", "Middle", "Most recent"]
    assert [sample["fighter1_odds"] for sample in result["samples"]] == [-125, -135, -145]


def test_get_sampled_odds_history_backfills_estimated_samples_when_only_one_snapshot(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    store_path = odds_dir / "the_odds_api_events.json"
    odds_dir.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_key": "the_odds_api|2026-05-30",
                        "event_name": "MMA Card · 2026-05-30",
                        "event_date": "2026-05-30",
                        "fights": [
                            {
                                "fight_key": odds_service.fight_key("Loma Lookboonmee", "Jaqueline Amorim"),
                                "fighter1": "Loma Lookboonmee",
                                "fighter2": "Jaqueline Amorim",
                                "odds_history": [
                                    {
                                        "captured_at": "2026-05-17T17:05:03Z",
                                        "fighter1_odds": 105,
                                        "fighter2_odds": -125,
                                        "fighter1_prob": 0.4878,
                                        "fighter2_prob": 0.5556,
                                        "bookmaker": "DraftKings",
                                        "last_update": "2026-05-17T17:05:03Z",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)

    result = odds_service.get_sampled_odds_history(
        event_date="2026-05-30",
        fighter1="Loma Lookboonmee",
        fighter2="Jaqueline Amorim",
    )

    assert result is not None
    assert result["history_count"] == 3
    assert result["real_history_count"] == 1
    assert result["uses_estimated_samples"] is True
    assert [sample["label"] for sample in result["samples"]] == ["Estimated first", "Estimated middle", "Most recent"]
    assert result["samples"][-1]["fighter1_odds"] == 105
    assert result["samples"][0]["bookmaker"] == "Estimated"


def test_toggle_bet_placed_persists_and_clears(monkeypatch, tmp_path):
    odds_dir = tmp_path / "data" / "future_fight_odds"
    store_path = odds_dir / "the_odds_api_events.json"
    odds_dir.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_key": "the_odds_api|2026-05-30",
                        "event_name": "MMA Card · 2026-05-30",
                        "event_date": "2026-05-30",
                        "fights": [
                            {
                                "fight_key": odds_service.fight_key("Loma Lookboonmee", "Jaqueline Amorim"),
                                "fighter1": "Loma Lookboonmee",
                                "fighter2": "Jaqueline Amorim",
                                "fighter1_odds": 105,
                                "fighter2_odds": -125,
                                "odds_history": [
                                    {
                                        "captured_at": "2026-05-17T17:05:03Z",
                                        "fighter1_odds": 105,
                                        "fighter2_odds": -125,
                                        "fighter1_prob": 0.4878,
                                        "fighter2_prob": 0.5556,
                                        "bookmaker": "DraftKings",
                                        "last_update": "2026-05-17T17:05:03Z",
                                    }
                                ],
                                "bet_placed": None,
                            }
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(odds_service, "STORE_PATH", store_path)
    monkeypatch.setattr(odds_service, "_utc_now", lambda: datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc))

    placed = odds_service.toggle_bet_placed(
        event_date="2026-05-30",
        fighter1="Loma Lookboonmee",
        fighter2="Jaqueline Amorim",
        bet_fighter="Loma Lookboonmee",
        stake=40,
        custom_odds=150,
    )
    assert placed is not None
    assert placed["active"] is True
    assert placed["bet_placed"]["fighter"] == "Loma Lookboonmee"
    assert placed["bet_placed"]["stake"] == 40
    assert placed["bet_placed"]["bet_odds"] == 150
    assert placed["bet_placed"]["listed_odds"] == 105
    assert placed["bet_placed"]["opponent_listed_odds"] == -125

    cleared = odds_service.toggle_bet_placed(
        event_date="2026-05-30",
        fighter1="Loma Lookboonmee",
        fighter2="Jaqueline Amorim",
        bet_fighter="Loma Lookboonmee",
    )
    assert cleared is not None
    assert cleared["active"] is False
    assert cleared["bet_placed"] is None

    store = json.loads(store_path.read_text())
    assert store["events"][0]["fights"][0]["bet_placed"] is None


def test_predict_service_attaches_tracked_bet_for_the_odds_api_rows(monkeypatch):
    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "event_url": float("nan"),
                "fighter1": "Loma Lookboonmee",
                "fighter2": "Jaqueline Amorim",
                "fighter1_odds": 105,
                "fighter2_odds": -125,
                "fighter1_prob": 0.4878,
                "fighter2_prob": 0.5556,
                "source_type": "the_odds_api",
                "source_file": "the_odds_api_new_events.csv",
            },
        ]
    )

    monkeypatch.setattr(predict_service, "_resolve_fighter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        predict_service,
        "get_bet_placed_map",
        lambda: {
            (
                "2026-05-30",
                odds_service.fight_key("Loma Lookboonmee", "Jaqueline Amorim"),
            ): {
                "fighter": "Loma Lookboonmee",
                "stake": 40,
                "bet_odds": 150,
                "listed_odds": 105,
                "opponent_listed_odds": -125,
                "placed_at": "2026-05-17T18:00:00Z",
            }
        },
    )

    events_map, _ = predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=predict_service.pd.DataFrame(),
        cache={},
        session=object(),
        extractor=object(),
    )

    fight = events_map["the_odds_api|2026-05-30"]["fights"][0]
    assert fight["bet_placed"]["fighter"] == "Loma Lookboonmee"
    assert fight["bet_placed"]["stake"] == 40
    assert fight["bet_placed"]["bet_odds"] == 150
    assert fight["bet_placed"]["opponent_listed_odds"] == -125


def test_api_odds_history_returns_404_when_missing(monkeypatch):
    monkeypatch.setattr(events_router, "get_sampled_odds_history", lambda **_: None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            events_router.api_odds_history(
                fighter1="Song Yadong",
                fighter2="Deiveson Figueiredo",
                event_date="2026-05-30",
            )
        )

    assert exc.value.status_code == 404


def test_api_bets_groups_tracked_mma_card_bets(monkeypatch):
    monkeypatch.setattr(events_router, "EXTERNAL_BETS_PATH", Path("/tmp/definitely-missing-odds.csv"))
    monkeypatch.setattr(
        events_router,
        "get_events_data",
        lambda: [
            {
                "event_name": "MMA Card · 2026-05-30",
                "event_date": "2026-05-30",
                "source_type": "the_odds_api",
                "fights": [
                    {
                        "fighter1": "Song Yadong",
                        "fighter2": "Deiveson Figueiredo",
                        "f1_odds": 120,
                        "f2_odds": -140,
                        "winner": "Song Yadong",
                        "edge": 8.4,
                        "bet_placed": {
                            "fighter": "Song Yadong",
                            "opponent": "Deiveson Figueiredo",
                            "stake": 40,
                            "bet_odds": 150,
                            "listed_odds": 135,
                            "opponent_listed_odds": -160,
                            "placed_at": "2026-05-29T12:00:00Z",
                        },
                    },
                    {
                        "fighter1": "Carlos Prates",
                        "fighter2": "Michael Morales",
                        "f1_odds": 110,
                        "f2_odds": -130,
                        "winner": None,
                        "edge": 5.1,
                        "bet_placed": {
                            "fighter": "Michael Morales",
                            "opponent": "Carlos Prates",
                            "stake": 30,
                            "bet_odds": -120,
                            "listed_odds": -120,
                            "opponent_listed_odds": 105,
                            "placed_at": "2026-05-29T12:05:00Z",
                        },
                    },
                ],
            },
            {
                "event_name": "UFC 330",
                "event_date": "2026-06-06",
                "source_type": "csv",
                "fights": [],
            },
        ],
    )

    payload = asyncio.run(events_router.api_bets())

    assert len(payload) == 1
    card = payload[0]
    assert card["event_name"] == "MMA Card · 2026-05-30"
    assert card["bet_count"] == 2
    assert card["settled_count"] == 1
    assert card["wins"] == 1
    assert card["losses"] == 0
    assert card["pending_count"] == 1
    assert card["total_risk"] == 40.0
    assert card["total_pnl"] == 60.0
    assert card["roi"] == 150.0
    assert card["bets"][0]["bet"]["won"] is True
    assert card["bets"][0]["bet"]["odds"] == 150
    assert card["bets"][0]["bet"]["current_odds"] == 120
    assert card["bets"][1]["bet"]["settled"] is False
    assert card["bets"][1]["bet"]["current_odds"] == -130


def test_api_bets_imports_external_csv_without_touching_tracked_store(monkeypatch, tmp_path):
    external_csv = tmp_path / "odds.csv"
    external_csv.write_text(
        "\t".join(
            [
                "date",
                "event",
                "bet_on",
                "type",
                "model_prob",
                "market_odds",
                "oppenent_odds",
                "closing_odds",
                "edge",
                "manual_confidence",
                "stake",
                "multiplier",
                "PNL",
                "Notes",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "May 2nd, 2026",
                "UFC FIGHT NIGHT: DELLA MADDALENA VS PRATES",
                "Cameron Rowston",
                "favorite model",
                "67.4",
                "-175",
                "150",
                "",
                "6",
                "7",
                "100",
                "1",
                "",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(events_router, "EXTERNAL_BETS_PATH", external_csv)
    db_fight = SimpleNamespace(
        id=1,
        event=SimpleNamespace(name="UFC Fight Night: Della Maddalena vs. Prates", date="May 02, 2026"),
        fighter_1=SimpleNamespace(name="Cam Rowston"),
        fighter_2=SimpleNamespace(name="Robert Bryczek"),
        fighter_1_id=11,
        fighter_2_id=12,
        winner_id=11,
        result="fighter_1",
        method="KO/TKO",
        round_finished=2,
    )

    class _FakeQuery:
        def __init__(self, fights):
            self._fights = fights

        def join(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._fights

    class _FakeSession:
        def query(self, *args, **kwargs):
            return _FakeQuery([db_fight])

        def close(self):
            return None

    monkeypatch.setattr(events_router, "_get_session", lambda: _FakeSession())
    monkeypatch.setattr(events_router, "_resolve_fighter", lambda session, name: SimpleNamespace(id=11, name="Cam Rowston"))
    monkeypatch.setattr(
        events_router,
        "get_events_data",
        lambda: [],
    )

    payload = asyncio.run(events_router.api_bets())

    assert len(payload) == 1
    card = payload[0]
    assert card["event_name"] == "UFC Fight Night: Della Maddalena vs. Prates"
    assert card["settled_count"] == 1
    assert card["wins"] == 1
    assert card["roi"] == 57.1
    assert card["bets"][0]["bet"]["fighter"] == "Cameron Rowston"
    assert card["bets"][0]["bet"]["opponent"] == "Robert Bryczek"
    assert card["bets"][0]["bet_type"] == "favorite model"
    assert card["bets"][0]["source_label"] == "Imported from /tmp/odds.csv"
    assert card["bets"][0]["winner"] == "Cam Rowston"
    assert card["event_date"] == "2026-05-02"


def test_event_match_score_handles_loose_event_labels():
    assert events_router._event_match_score("UFC 328", "UFC 328: Chimaev vs. Strickland") > 1.0
    assert events_router._event_match_score("UFC Vegas 116", "UFC Fight Night: Sterling vs. Zalal") == 0.0


def test_future_prediction_cache_key_rolls_daily(monkeypatch):
    monkeypatch.setattr(predict_service, "_now", _fixed_now)

    key = predict_service._cache_key_for_prediction(
        "ilia topuria_vs_islam makhachev",
        as_of=datetime(2026, 6, 2, 0, 0, 0),
        event_date="2026-06-02",
    )

    assert key == "v2|ilia topuria_vs_islam makhachev|future|2026-06-02|2026-05-18"


def test_run_prediction_loop_ignores_previous_day_future_cache(monkeypatch):
    monkeypatch.setattr(predict_service, "_now", _fixed_now)
    monkeypatch.setattr(predict_service, "get_bet_placed_map", lambda: {})

    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "MMA Card · 2026-06-02",
                "event_date": "2026-06-02",
                "event_url": "",
                "fighter1": "Islam Makhachev",
                "fighter2": "Ilia Topuria",
                "fighter1_odds": -234,
                "fighter2_odds": 199,
                "fighter1_prob": 0.7006,
                "fighter2_prob": 0.3344,
                "source_type": "the_odds_api",
                "source_file": "the_odds_api_new_events.csv",
            }
        ]
    )

    stale_key = "v2|ilia topuria_vs_islam makhachev|future|2026-06-02|2026-05-17"
    cache = {
        stale_key: {
            "model_prob_f1": 0.5329,
            "model_source": "general",
            "f1_db_name": "Islam Makhachev",
            "f2_db_name": "Ilia Topuria",
            "f1_fight_count": 18,
            "f2_fight_count": 9,
            "is_wmma": False,
        }
    }

    monkeypatch.setattr(
        predict_service,
        "_resolve_fighter",
        lambda _session, name: type("Fighter", (), {"id": 1 if "Islam" in name else 2, "name": name})(),
    )
    monkeypatch.setattr(
        predict_service,
        "_score_row",
        lambda *_args, **_kwargs: {"model_prob_f1": 0.512, "model_source": "general"},
    )
    monkeypatch.setattr(
        predict_service,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 18 if fighter_id == 1 else 9,
    )
    monkeypatch.setattr(
        predict_service,
        "_is_wmma",
        lambda *_args, **_kwargs: False,
    )

    events_map, cache_dirty = predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=predict_service.pd.DataFrame(),
        cache=cache,
        session=object(),
        extractor=object(),
    )

    fight = events_map["the_odds_api|2026-06-02"]["fights"][0]
    assert cache_dirty is True
    assert fight["model_prob_f1"] == 51.2
    assert cache["v2|ilia topuria_vs_islam makhachev|future|2026-06-02|2026-05-18"]["model_prob_f1"] == 0.512
    assert stale_key in cache


def test_run_prediction_loop_keeps_past_event_date_for_unresolved_fight(monkeypatch):
    monkeypatch.setattr(predict_service, "_now", _fixed_now)
    monkeypatch.setattr(predict_service, "get_bet_placed_map", lambda: {})

    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "UFC Vegas 117",
                "event_date": "May 17th",
                "event_url": "https://www.bestfightodds.com/events/ufc-vegas-117-4178",
                "fighter1": "Daniel Santos",
                "fighter2": "Doo Ho Choi",
                "fighter1_odds": -132,
                "fighter2_odds": 108,
                "fighter1_prob": 0.542,
                "fighter2_prob": 0.458,
                "source_type": "user_added",
                "source_file": "https_www_bestfightodds_com_events_ufc_vegas_117_4178.json",
            }
        ]
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        predict_service,
        "_resolve_fighter",
        lambda _session, name: type("Fighter", (), {"id": 1 if "Daniel" in name else 2, "name": name})(),
    )

    def fake_score_row(_session, _extractor, _f1_id, _f2_id, _mkt_prob, *, as_of_date=None):
        captured["as_of_date"] = as_of_date
        return {"model_prob_f1": 0.376, "model_source": "general"}

    monkeypatch.setattr(predict_service, "_score_row", fake_score_row)
    monkeypatch.setattr(
        predict_service,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 6 if fighter_id == 1 else 10,
    )
    monkeypatch.setattr(predict_service, "_is_wmma", lambda *_args, **_kwargs: False)

    predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=predict_service.pd.DataFrame(),
        cache={},
        session=object(),
        extractor=object(),
    )

    assert captured["as_of_date"] == datetime(2026, 5, 17, 0, 0, 0)


def test_run_prediction_loop_keeps_past_event_date_for_completed_fight(monkeypatch):
    monkeypatch.setattr(predict_service, "_now", _fixed_now)
    monkeypatch.setattr(predict_service, "get_bet_placed_map", lambda: {})

    odds_df = predict_service.pd.DataFrame(
        [
            {
                "event_name": "UFC Vegas 117",
                "event_date": "May 17th",
                "event_url": "https://www.bestfightodds.com/events/ufc-vegas-117-4178",
                "fighter1": "Daniel Santos",
                "fighter2": "Doo Ho Choi",
                "fighter1_odds": -132,
                "fighter2_odds": 108,
                "fighter1_prob": 0.542,
                "fighter2_prob": 0.458,
                "source_type": "user_added",
                "source_file": "https_www_bestfightodds_com_events_ufc_vegas_117_4178.json",
            }
        ]
    )
    outcomes = predict_service.pd.DataFrame(
        [
            {
                "fighter1": "Daniel Santos",
                "fighter2": "Doo Ho Choi",
                "winner": "Doo Ho Choi",
                "method": "KO/TKO",
                "round": "2",
                "norm_key": predict_service._fight_key("Daniel Santos", "Doo Ho Choi"),
            }
        ]
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        predict_service,
        "_resolve_fighter",
        lambda _session, name: type("Fighter", (), {"id": 1 if "Daniel" in name else 2, "name": name})(),
    )

    def fake_score_row(_session, _extractor, _f1_id, _f2_id, _mkt_prob, *, as_of_date=None):
        captured["as_of_date"] = as_of_date
        return {"model_prob_f1": 0.376, "model_source": "general"}

    monkeypatch.setattr(predict_service, "_score_row", fake_score_row)
    monkeypatch.setattr(
        predict_service,
        "_fight_count_as_of",
        lambda _session, fighter_id, _as_of: 6 if fighter_id == 1 else 10,
    )
    monkeypatch.setattr(predict_service, "_is_wmma", lambda *_args, **_kwargs: False)

    predict_service._run_prediction_loop(
        odds_df=odds_df,
        outcomes=outcomes,
        cache={},
        session=object(),
        extractor=object(),
    )

    assert captured["as_of_date"] == datetime(2026, 5, 17, 0, 0, 0)
