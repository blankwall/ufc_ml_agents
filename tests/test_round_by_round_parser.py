from pathlib import Path

from bs4 import BeautifulSoup

from scrapers.event_scraper import EventScraper

EVENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "events"
MULTI_ROUND_PAGE = EVENTS_DIR / "fight_8bee5192baec71d8.html"
ONE_ROUND_PAGE = EVENTS_DIR / "fight_a26ea74f2e908842.html"


def _scraper():
    return EventScraper(config_path="config/config.yaml")


def _soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")


def test_one_round_cached_page_parses_single_round():
    rounds = _scraper()._extract_round_by_round(_soup(ONE_ROUND_PAGE))

    assert len(rounds) == 1
    r1 = rounds[0]
    assert r1["round"] == 1
    # Round totals map by position (KD column), fighter_1 landed a knockdown.
    assert r1["fighter_1"]["totals"]["knockdowns"] == "1"
    assert r1["fighter_2"]["totals"]["knockdowns"] == "0"
    # Sig-strike breakdown merged into the same round object.
    assert r1["fighter_1"]["significant_strikes"]["head_strikes"] == "18 of 43"


def test_multi_round_cached_page_parses_distinct_rounds():
    rounds = _scraper()._extract_round_by_round(_soup(MULTI_ROUND_PAGE))

    assert [r["round"] for r in rounds] == [1, 2, 3]
    # Distinct per-round sig-strike volume for fighter_2 (Jose Delgado): 23/30/41.
    f2_sig = [r["fighter_2"]["totals"]["sig_strikes"] for r in rounds]
    assert f2_sig == ["23 of 46", "30 of 62", "41 of 61"]


def test_takedowns_column_not_collapsed_by_duplicate_header():
    # UFCStats _rnd totals table mislabels the takedowns column as "Td %".
    # Position-based mapping must keep takedowns and takedown_pct distinct.
    rounds = _scraper()._extract_round_by_round(_soup(MULTI_ROUND_PAGE))

    r1_f1 = rounds[0]["fighter_1"]["totals"]
    assert r1_f1["takedowns"] == "1 of 1"
    assert r1_f1["takedown_pct"] == "100%"


def test_totals_and_sig_merge_into_one_object_per_round():
    rounds = _scraper()._extract_round_by_round(_soup(MULTI_ROUND_PAGE))

    for r in rounds:
        for slot in ("fighter_1", "fighter_2"):
            assert set(r[slot].keys()) == {"totals", "significant_strikes"}
            assert r[slot]["totals"], f"missing totals for {slot} round {r['round']}"
            assert r[slot]["significant_strikes"], f"missing sig for {slot} round {r['round']}"
    # Control-time preserved as raw UFCStats string.
    assert rounds[0]["fighter_1"]["totals"]["control_time"] == "0:23"


def test_no_rnd_tables_returns_empty_list():
    soup = BeautifulSoup("<html><body><p>no tables here</p></body></html>", "lxml")
    assert _scraper()._extract_round_by_round(soup) == []


def test_scrape_fight_details_includes_round_by_round(monkeypatch, tmp_path):
    scraper = _scraper()
    html = MULTI_ROUND_PAGE.read_text(encoding="utf-8")

    # Force the fetch path to return our fixture and write any cache to a temp dir.
    monkeypatch.setattr(
        "scrapers.event_scraper.fetch_ufcstats_html",
        lambda *args, **kwargs: html,
    )
    scraper.config["scraping"]["cache_enabled"] = False
    scraper.cache_dir = tmp_path

    details = scraper.scrape_fight_details(
        "http://ufcstats.com/fight-details/nonexistenthash"
    )

    assert details is not None
    assert isinstance(details["round_by_round"], list)
    assert [r["round"] for r in details["round_by_round"]] == [1, 2, 3]
