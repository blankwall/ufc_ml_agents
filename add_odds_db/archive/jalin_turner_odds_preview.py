#!/usr/bin/env python3
"""
Preview opening/closing odds for Jalin Turner from BestFightOdds.

GOAL (step 1, read-only):
  - Fetch Jalin Turner's fighter page on BestFightOdds
  - Parse the odds-history table directly (no graphs API)
  - For each matchup row where Jalin Turner appears:
        event_slug, event_label, opponent_name,
        opening_american_odds, closing_american_odds (final cell)
  - Print a tidy table so you can manually validate before we touch the DB.

This relies on the fighter odds history table structure seen at:
  https://www.bestfightodds.com/fighters/Jalin-Turner-6854
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent

JALIN_TURNER_URL = "https://www.bestfightodds.com/fighters/Jalin-Turner-6854"
JALIN_NAME = "Jalin Turner"


def main() -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        ),
        "Referer": "https://www.bestfightodds.com/",
    }

    print(f"Fetching fighter page: {JALIN_TURNER_URL}")
    resp = requests.get(JALIN_TURNER_URL, headers=headers)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="team-stats-table")
    if table is None:
        print("Could not find odds history table on fighter page.")
        return

    tbody = table.find("tbody")
    if tbody is None:
        print("Table has no tbody.")
        return

    rows: list[dict] = []
    current_event_slug: str | None = None
    current_event_label: str | None = None

    # Iterate rows in order; event-header rows set the context, main-row pairs
    # contain fighter + opponent.
    for tr in tbody.find_all("tr", recursive=False):
        classes = tr.get("class", [])
        if "event-header" in classes:
            # e.g. <a href="/events/ufc-3895">UFC</a> Dec 7th 2025
            a = tr.find("a", href=re.compile(r"^/events/"))
            if a is not None:
                href = a.get("href", "")
                slug = href.rstrip("/").split("/")[-1]
                label = tr.get_text(" ", strip=True)

                # Ignore future/unconfirmed sections and non-UFC orgs.
                if not slug.startswith("ufc-"):
                    current_event_slug = None
                    current_event_label = None
                elif "Unconfirmed Fights" in label or "Future Events" in label:
                    current_event_slug = None
                    current_event_label = None
                else:
                    current_event_slug = slug
                    current_event_label = label
        elif "main-row" in classes:
            # Check if this is Jalin's row
            link = tr.find("a", href=re.compile(r"/fighters/"))
            if not link:
                continue
            fighter_name = link.get_text(strip=True)
            if fighter_name != JALIN_NAME:
                continue

            # Require a valid UFC event context
            if not current_event_slug or not current_event_slug.startswith("ufc-"):
                continue

            # Use first/last moneyline cells for open/close
            money_tds = tr.find_all("td", class_="moneyline")
            if len(money_tds) < 2:
                continue
            open_str = money_tds[0].get_text(strip=True)
            close_str = money_tds[-1].get_text(strip=True)

            # Opponent is in the next main-row
            opp_tr = tr.find_next_sibling("tr", class_="main-row")
            opponent_name = None
            if opp_tr is not None:
                opp_link = opp_tr.find("a", href=re.compile(r"/fighters/"))
                if opp_link is not None:
                    opponent_name = opp_link.get_text(strip=True)

            rows.append(
                {
                    "event_slug": current_event_slug,
                    "event_label": current_event_label,
                    "fighter": JALIN_NAME,
                    "opponent": opponent_name,
                    "opening_american": open_str,
                    "closing_american": close_str,
                }
            )

    if not rows:
        print("\nNo Jalin Turner matchups with graph data found.")
        return

    df = pd.DataFrame(rows)

    # Deduplicate per (event_slug, fighter): keep the row with the
    # most odds movement (largest |close_prob - open_prob|).
    def _implied_prob(american: str) -> float:
        try:
            v = int(str(american).replace("+", "").replace(" ", ""))
            return 100.0 / (v + 100.0) if v > 0 else abs(v) / (abs(v) + 100.0)
        except (ValueError, ZeroDivisionError):
            return 0.5

    df["_open_prob"]  = df["opening_american"].apply(_implied_prob)
    df["_close_prob"] = df["closing_american"].apply(_implied_prob)
    df["_movement"]   = (df["_close_prob"] - df["_open_prob"]).abs()
    df = (
        df.sort_values("_movement", ascending=False)
          .drop_duplicates(subset=["event_slug", "fighter"])
          .sort_index()
          .drop(columns=["_open_prob", "_close_prob", "_movement"])
          .reset_index(drop=True)
    )
    print("\nSummary table (Jalin Turner – opening/closing odds by fight):\n")
    print(
        df[
            [
                "event_slug",
                "event_label",
                "opponent",
                "opening_american",
                "closing_american",
            ]
        ].to_string(index=False)
    )

    out_path = ROOT / "analysis" / "jalin_turner_odds_preview.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote CSV preview → {out_path}")


if __name__ == "__main__":
    main()

