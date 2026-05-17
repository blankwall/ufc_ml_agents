#!/usr/bin/env python3
"""
Generic fighter odds preview from BestFightOdds.

Given a fighter name, this script:
  - Searches BestFightOdds to find the fighter's odds-history page
  - Parses the fighter's odds-history table
  - Extracts, for each matchup row:
        event_slug, event_label, opponent_name,
        opening_american_odds, closing_american_odds (final closing cell)
  - Prints a tidy table and writes it to analysis/<slug>_odds_preview.csv

This is a generalized version of the Jalin Turner preview script.
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent


BASE_URL = "https://www.bestfightodds.com"


def build_search_url(name: str) -> str:
    q = urllib.parse.quote_plus(name)
    return f"{BASE_URL}/search?query={q}"


def find_fighter_url(fighter_name: str) -> str:
    """
    Use the BestFightOdds search endpoint to find the fighter odds-history URL.

    Strategy:
      - Request /search?query=<fighter_name>
      - Look for <a href="/fighters/..."> tags
      - Prefer exact text match on fighter_name (case-insensitive)
      - Fallback to the first fighter link if no exact match
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        )
    }

    search_url = build_search_url(fighter_name)
    resp = requests.get(search_url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=re.compile(r"^/fighters/"))
    if not links:
        raise RuntimeError(f"No fighter links found on search page for '{fighter_name}' ({search_url})")

    target_lower = fighter_name.strip().lower()
    best = None
    for a in links:
        text = a.get_text(strip=True)
        if text.lower() == target_lower:
            best = a
            break
    if best is None:
        best = links[0]

    href = best.get("href", "")
    if not href.startswith("/fighters/"):
        raise RuntimeError(f"Unexpected fighter href '{href}' on search page")
    return urllib.parse.urljoin(BASE_URL, href)


def parse_fighter_odds_page(fighter_url: str, fighter_name: str) -> pd.DataFrame:
    """
    Parse a fighter odds-history page and return a DataFrame with:
      event_slug, event_label, fighter, opponent, opening_american, closing_american
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
        ),
        "Referer": BASE_URL + "/",
    }

    resp = requests.get(fighter_url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="team-stats-table")
    if table is None:
        raise RuntimeError("Could not find odds history table ('team-stats-table') on fighter page.")

    tbody = table.find("tbody")
    if tbody is None:
        raise RuntimeError("Odds history table has no <tbody>.")

    rows: list[dict] = []
    current_event_slug: str | None = None
    current_event_label: str | None = None

    for tr in tbody.find_all("tr", recursive=False):
        classes = tr.get("class", [])
        if "event-header" in classes:
            # Example: <a href="/events/ufc-3895">UFC</a> Dec 7th 2025
            a = tr.find("a", href=re.compile(r"^/events/"))
            if a is not None:
                href = a.get("href", "")
                slug = href.rstrip("/").split("/")[-1]
                label = tr.get_text(" ", strip=True)

                # Drop future/unconfirmed sections entirely.
                # BestFightOdds uses slugs like 'future-events-197' and
                # 'unconfirmed-fights-3781' plus labels 'Future Events', 'Unconfirmed Fights'.
                # Accept "ufc" or "ufc-xxx" as valid UFC event slugs
                print("Slug: ",slug)
                if not (slug.startswith("ufc-") or slug == "ufc"):
                    current_event_slug = None
                    current_event_label = None
                elif "Unconfirmed Fights" in label or "Future Events" in label:
                    current_event_slug = None
                    current_event_label = None
                else:
                    current_event_slug = slug
                    current_event_label = label
        elif "main-row" in classes:
            link = tr.find("a", href=re.compile(r"/fighters/"))
            if not link:
                continue
            name = link.get_text(strip=True)
            if name != fighter_name:
                continue

            # Skip if we don't currently have a valid UFC event context
            # Accept "ufc" (for unnumbered events) or "ufc-xxx" (for numbered events)
            if not current_event_slug or not (current_event_slug.startswith("ufc-") or current_event_slug == "ufc"):
                continue

            # Use first/last moneyline cells for open/close
            money_tds = tr.find_all("td", class_="moneyline")
            if len(money_tds) < 2:
                continue
            open_str = money_tds[0].get_text(strip=True)
            close_str = money_tds[-1].get_text(strip=True)

            # Opponent is in the immediate next sibling row (no class, not main-row)
            opponent_name = None
            next_tr = tr.find_next_sibling("tr")
            if next_tr is not None:
                opp_link = next_tr.find("a", href=re.compile(r"/fighters/"))
                if opp_link is not None:
                    opponent_name = opp_link.get_text(strip=True)

            rows.append(
                {
                    "event_slug": current_event_slug,
                    "event_label": current_event_label,
                    "fighter": fighter_name,
                    "opponent": opponent_name,
                    "opening_american": open_str,
                    "closing_american": close_str,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # No deduplication - return all rows and let the matching logic in
    # insert_fighter_odds.py determine which rows correspond to actual DB fights.
    # BFO sometimes lists multiple potential matchups for the same event,
    # and we want to match against all of them.
    return df.reset_index(drop=True)


def safe_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "fighter"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview BestFightOdds opening/closing odds for a fighter.")
    parser.add_argument("fighter_name", help="Fighter name as in the UFC DB (e.g. 'Jalin Turner').")
    parser.add_argument(
        "--fighter-url",
        help="Optional full BestFightOdds fighter URL; if omitted, will search for the fighter.",
    )
    args = parser.parse_args()

    fighter_name = args.fighter_name

    if args.fighter_url:
        fighter_url = args.fighter_url
    else:
        print(f"Searching BestFightOdds for fighter: {fighter_name}")
        fighter_url = find_fighter_url(fighter_name)

    print(f"Using fighter URL: {fighter_url}\n")

    df = parse_fighter_odds_page(fighter_url, fighter_name)

    if df.empty:
        print("No odds rows found for this fighter.")
        return

    print(f"Found {len(df)} fights for {fighter_name}.\n")
    print("Summary table (opening/closing odds by fight):\n")
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

    out_name = f"{safe_slug(fighter_name)}_odds_preview.csv"
    out_path = ROOT / "analysis" / out_name
    df.to_csv(out_path, index=False)
    print(f"\nWrote CSV preview → {out_path}")


if __name__ == "__main__":
    main()

