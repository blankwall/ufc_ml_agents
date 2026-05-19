from scrapers.sherdog_scraper import SherdogScraper


def _build_scraper():
    return SherdogScraper(config_path="config/config.yaml")


def test_parse_sherdog_fighter_page_extracts_bio_and_history():
    html = """
    <html>
      <head>
        <meta itemprop="name" content="Yi Sak Lee" />
      </head>
      <body>
        <div class="module bio_fighter vcard">
          <div class="fighter-info">
            <div class="fighter-right">
              <div class="fighter-line1">
                <div class="fighter-flag-social">
                  <div class="fighter-nationality">
                    <span class="item birthplace">
                      <strong itemprop="nationality">South Korea</strong>
                      <span itemprop="address" itemscope itemtype="http://schema.org/PostalAddress" class="adr">
                        <span itemprop="addressLocality" class="locality">Seoul</span>
                      </span>
                    </span>
                  </div>
                </div>
              </div>
              <div class="fighter-data">
                <div class="bio-holder">
                  <table>
                    <tr>
                      <td>AGE</td>
                      <td><b>26</b> <em>/</em> <span itemprop="birthDate">Jan 11, 2000</span></td>
                    </tr>
                    <tr>
                      <td>HEIGHT</td>
                      <td><b itemprop="height">6'0"</b> <em>/</em> 182.88 cm</td>
                    </tr>
                    <tr>
                      <td>WEIGHT</td>
                      <td><b itemprop="weight">185 lbs</b> <em>/</em> 83.91 kg</td>
                    </tr>
                  </table>
                  <div class="association-class">
                    ASSOCIATION<br />
                    <a class="association" href="/stats/fightfinder?association=Korean+Top+Team">Korean Top Team</a>
                    <br /><br />
                    CLASS<br />
                    <a href="/stats/fightfinder?weightclass=Middleweight">Middleweight</a>
                  </div>
                </div>
                <div class="winsloses-holder">
                  <div class="wins">
                    <div class="winloses win"><span>Wins</span><span>8</span></div>
                    <div class="meter-title">KO / TKO</div>
                    <div class="meter">
                      <div class="pl">4</div>
                      <div class="pm"></div>
                      <div class="pr">50%</div>
                    </div>
                    <div class="meter-title">SUBMISSIONS</div>
                    <div class="meter">
                      <div class="pl">3</div>
                      <div class="pm"></div>
                      <div class="pr">38%</div>
                    </div>
                  </div>
                  <div class="loses">
                    <div class="winloses lose"><span>Losses</span><span>1</span></div>
                    <div class="meter-title">SUBMISSIONS</div>
                    <div class="meter">
                      <div class="pl">1</div>
                      <div class="pm"></div>
                      <div class="pr">100%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="module fight_history">
          <table class="new_table fighter">
            <tr class="table_head">
              <td>Result</td><td>Fighter</td><td>Event</td><td>Method</td><td>R</td><td>Time</td>
            </tr>
            <tr>
              <td><span class="final_result win">win</span></td>
              <td><a href="/fighter/Daichi-Henry-Mikami-392951">Daichi Henry Mikami</a></td>
              <td>
                <a href="/events/Heat-Heat-57-109307"><span itemprop="award">Heat - Heat 57</span></a>
                <br /><span class="sub_line">Sep / 20 / 2025</span>
              </td>
              <td class="winby">
                <b>Submission (Rear-Naked Choke)</b>
                <br /><span class="sub_line"><a href="/referee/Kosuke-Umeda-695">Kosuke Umeda</a></span>
              </td>
              <td>3</td>
              <td>4:59</td>
            </tr>
            <tr>
              <td><span class="final_result loss">loss</span></td>
              <td><a href="/fighter/Agilan-Thani-153265">Agilan Thani</a></td>
              <td>
                <a href="/events/BTC-Breakthrough-Combat-2-106131">BTC - Breakthrough Combat 2</a>
                <br /><span class="sub_line">Dec / 25 / 2024</span>
              </td>
              <td class="winby">
                <b>Submission (Rear-Naked Choke)</b>
                <br /><span class="sub_line"></span>
              </td>
              <td>2</td>
              <td>2:33</td>
            </tr>
          </table>
        </div>
      </body>
    </html>
    """

    fighter = _build_scraper().parse_fighter_html(
        html,
        fighter_url="https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
    )

    assert fighter["fighter_id"] == "390123"
    assert fighter["source"] == "sherdog"
    assert fighter["name"] == "Yi Sak Lee"
    assert fighter["nationality"] == "South Korea"
    assert fighter["birthplace"] == "Seoul"
    assert fighter["association"] == "Korean Top Team"
    assert fighter["weight_class"] == "Middleweight"
    assert fighter["age"] == 26
    assert fighter["date_of_birth"] == "Jan 11, 2000"
    assert fighter["height_cm"] == 182.88
    assert fighter["weight_lbs"] == 185
    assert fighter["method_breakdown"]["wins"]["total"] == 8
    assert fighter["method_breakdown"]["wins"]["ko_tko"]["count"] == 4
    assert fighter["method_breakdown"]["wins"]["ko_tko"]["percentage"] == 0.5
    assert fighter["method_breakdown"]["losses"]["submissions"]["count"] == 1
    assert fighter["fight_history"] == [
        {
            "result": "win",
            "opponent": "Daichi Henry Mikami",
            "opponent_url": "https://www.sherdog.com/fighter/Daichi-Henry-Mikami-392951",
            "event": "Heat - Heat 57",
            "event_url": "https://www.sherdog.com/events/Heat-Heat-57-109307",
            "date": "2025-09-20",
            "method": "Submission (Rear-Naked Choke)",
            "referee": "Kosuke Umeda",
            "round": 3,
            "time": "4:59",
        },
        {
            "result": "loss",
            "opponent": "Agilan Thani",
            "opponent_url": "https://www.sherdog.com/fighter/Agilan-Thani-153265",
            "event": "BTC - Breakthrough Combat 2",
            "event_url": "https://www.sherdog.com/events/BTC-Breakthrough-Combat-2-106131",
            "date": "2024-12-25",
            "method": "Submission (Rear-Naked Choke)",
            "referee": "",
            "round": 2,
            "time": "2:33",
        },
    ]


def test_normalize_sherdog_date_falls_back_when_unparseable():
    assert _build_scraper()._normalize_sherdog_date("TBA") == "TBA"


def test_parse_search_results_extracts_fighter_matches():
    html = """
    <html>
      <body>
        <h3>FIGHTER RESULTS</h3>
        <table class="new_table fightfinder_result">
          <tr class="table_head">
            <td>Image</td><td>Fighter</td><td>Nickname</td><td>Height</td><td>Weight</td><td>Association</td>
          </tr>
          <tr>
            <td><img src="/image_crop/100/100/_images/fighter_small_default.jpg" /></td>
            <td><a href="/fighter/Yi-Sak-Lee-390123">Yi Sak Lee</a></td>
            <td></td>
            <td>6'0" (1.83 m)</td>
            <td>185 lbs (83.91 kg)</td>
            <td>Korean Top Team</td>
          </tr>
        </table>
      </body>
    </html>
    """

    results = _build_scraper().parse_search_results(html)

    assert results == [
        {
            "name": "Yi Sak Lee",
            "url": "https://www.sherdog.com/fighter/Yi-Sak-Lee-390123",
            "fighter_id": "390123",
            "nickname": None,
            "height": "6'0\" (1.83 m)",
            "weight": "185 lbs (83.91 kg)",
            "association": "Korean Top Team",
        }
    ]
