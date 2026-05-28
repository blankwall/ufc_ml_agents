from bs4 import BeautifulSoup

from scrapers.event_scraper import EventScraper
from scrapers.ufcstats_challenge import fetch_ufcstats_html


def _build_scraper():
    return EventScraper(config_path="config/config.yaml")


def test_extract_fight_outcome_recognizes_no_contest_from_details_page():
    soup = BeautifulSoup(
        """
        <div class="b-fight-details__person">
          <i class="b-fight-details__person-status b-fight-details__person-status_style_gray">NC</i>
          <a class="b-link b-link_style_black" href="http://ufcstats.com/fighter-details/f1">Fighter One</a>
        </div>
        <div class="b-fight-details__person">
          <i class="b-fight-details__person-status b-fight-details__person-status_style_gray">NC</i>
          <a class="b-link b-link_style_black" href="http://ufcstats.com/fighter-details/f2">Fighter Two</a>
        </div>
        <p class="b-fight-details__text"></p>
        """,
        "lxml",
    )

    outcome = _build_scraper()._extract_fight_outcome(soup)

    assert outcome["fighter_1_result"] == "NC"
    assert outcome["fighter_2_result"] == "NC"
    assert outcome["winner"] == "no_contest"


def test_get_all_event_links_completed_parses_date_from_span(monkeypatch):
    html = """
    <table>
      <tr class="b-statistics__table-row">
        <th>Header</th>
      </tr>
      <tr class="b-statistics__table-row">
        <td class="b-statistics__table-col">
          <i class="b-statistics__table-content">
            <a class="b-link b-link_style_black" href="http://ufcstats.com/event-details/evt-123">
              UFC Test Card
            </a>
            <span class="b-statistics__date">May 16, 2026</span>
          </i>
        </td>
        <td class="b-statistics__table-col b-statistics__table-col_style_big-top-padding">
          Las Vegas, Nevada, USA
        </td>
      </tr>
    </table>
    """

    class _FakeResponse:
        def __init__(self, text):
            self.text = text
            self.content = text.encode("utf-8")

        def raise_for_status(self):
            return None

    scraper = _build_scraper()
    monkeypatch.setattr(scraper.session, "get", lambda *args, **kwargs: _FakeResponse(html))
    monkeypatch.setattr("scrapers.event_scraper.time.sleep", lambda *_args, **_kwargs: None)

    events = scraper.get_all_event_links(completed_only=True, max_pages=1)

    assert events == [
        {
            "name": "UFC Test Card",
            "url": "http://ufcstats.com/event-details/evt-123",
            "event_id": "evt-123",
            "date": "May 16, 2026",
            "location": "Las Vegas, Nevada, USA",
        }
    ]


def test_fetch_ufcstats_html_solves_browser_challenge():
    challenge_html = """
    <html><body>
    <p>Checking your browser</p>
    <script>
    var nonce="abc123",
        target=new Array(1+1).join('0');
    var xhr=new XMLHttpRequest();
    xhr.open('POST',"/__c",true);
    </script>
    </body></html>
    """
    expected_html = "<html><body>UFC 325</body></html>"

    class _FakeResponse:
        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            return None

    class _FakeSession:
        def __init__(self):
            self.get_calls = 0
            self.post_data = None
            self.post_url = None

        def get(self, url, timeout):
            self.get_calls += 1
            return _FakeResponse(challenge_html if self.get_calls == 1 else expected_html)

        def post(self, url, data, timeout, headers):
            self.post_url = url
            self.post_data = data
            return _FakeResponse(status_code=204)

    session = _FakeSession()

    html = fetch_ufcstats_html(session, "http://ufcstats.com/event-details/test", timeout=30)

    assert html == expected_html
    assert session.get_calls == 2
    assert session.post_url == "http://ufcstats.com/__c"
    assert session.post_data["nonce"] == "abc123"
    assert str(session.post_data["n"]).isdigit()
