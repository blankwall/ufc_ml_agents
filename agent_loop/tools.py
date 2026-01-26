from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from agent_loop.utils import run_cmd, write_json


def normalize_ufcstats_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("https://ufcstats.com/"):
        return "http://ufcstats.com/" + u[len("https://ufcstats.com/") :]
    return u


def fight_details_to_event_url(fight_url: str, *, timeout: int = 30) -> str:
    """
    Fetch a UFCStats fight-details page and extract the event-details URL.
    UFCStats fight pages typically include a link to /event-details/<id>.
    """
    fight_url = normalize_ufcstats_url(fight_url)
    resp = requests.get(fight_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Search any anchor for an event-details link
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/event-details/" in href:
            return normalize_ufcstats_url(href)

    # Fallback: regex scan full html
    m = re.search(r"http[s]?://ufcstats\.com/event-details/[a-z0-9]+", resp.text, re.I)
    if m:
        return normalize_ufcstats_url(m.group(0))

    raise RuntimeError(f"Could not find event-details link on fight page: {fight_url}")


def scrape_fight_details_json(fight_url: str, *, timeout: int = 30) -> Dict:
    """
    Minimal JSON scrape from fight-details page for planning context:
    - fighter ids + names
    - totals + sig strikes breakdown (as scraped strings)
    """
    fight_url = normalize_ufcstats_url(fight_url)
    resp = requests.get(fight_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Fighter sections (fight-details page)
    # Structure example:
    #   <div class="b-fight-details__person">
    #     <i class="b-fight-details__person-status ...">W</i>
    #     <a class="b-link b-fight-details__person-link" href=".../fighter-details/<id>">Name</a>
    #   </div>
    people = soup.select("div.b-fight-details__persons div.b-fight-details__person")
    fighters = []
    for p in people[:2]:
        status_el = p.select_one("i.b-fight-details__person-status")
        status = status_el.get_text(strip=True) if status_el else None

        link = p.select_one("a.b-fight-details__person-link[href]")
        if link is None:
            # Fallback: any link to fighter-details within this block
            link = p.select_one("a[href*='/fighter-details/'], a[href*='ufcstats.com/fighter-details/']")

        name = link.get_text(strip=True) if link else None
        href = link.get("href") if link else None
        href = normalize_ufcstats_url(href) if href else None
        fighter_id = None
        if href and "/fighter-details/" in href:
            fighter_id = href.split("/fighter-details/")[-1].strip("/")

        fighters.append(
            {
                "name": name,
                "ufcstats_id": fighter_id,
                "url": href,
                "status": status,
            }
        )

    return {
        "fight_details_url": fight_url,
        "fighters": fighters,
    }


def ingest_event_for_fight(
    repo_root: Path,
    fight_url: str,
    *,
    include_fight_stats: bool = True,
    validate: bool = True,
    validate_details: bool = True,
) -> str:
    """
    Resolve event URL from fight-details URL and ingest it using the existing populator.
    Returns the event URL.
    """
    event_url = fight_details_to_event_url(fight_url)
    cmd = [
        "python3",
        "scrapers/event_populator.py",
        "--event-url",
        event_url,
    ]
    if include_fight_stats:
        cmd.append("--include-fight-stats")
    if validate:
        cmd.append("--validate")
    if validate and validate_details:
        cmd.append("--validate-details")

    run_cmd(cmd, cwd=repo_root, check=True)
    return event_url


def run_xgboost_predict(
    repo_root: Path,
    *,
    fighter_1_ufcstats_id: str,
    fighter_2_ufcstats_id: str,
    model_name: str,
    out_path: Path,
) -> None:
    cmd = [
        "python3",
        "xgboost_predict.py",
        "--fighter-1",
        "F1",
        "--fighter-2",
        "F2",
        "--fighter-1-ufcstats-id",
        fighter_1_ufcstats_id,
        "--fighter-2-ufcstats-id",
        fighter_2_ufcstats_id,
        "--model-name",
        model_name,
        "--debug",
        "3",
    ]
    # We don't rely on stdout parsing; store raw output.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(cmd, cwd=repo_root, stdout_path=out_path, stderr_path=out_path.with_suffix(".stderr.txt"), check=True)


