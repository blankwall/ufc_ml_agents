"""
Sherdog fighter-page scraper.

Scrapes a single Sherdog fighter profile and returns basic bio data plus
fight-history rows that can be used for manual fighter backfills.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.parse import quote_plus

import requests
import yaml
from bs4 import BeautifulSoup
from loguru import logger


BASE_URL = "https://www.sherdog.com"


class SherdogScraper:
    """Scrapes fighter profile pages from Sherdog."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.rate_limit = float(self.config["scraping"]["rate_limit"]) if "rate_limit" in self.config["scraping"] else 1.0
        self.user_agent = self.config["scraping"]["user_agent"]
        self.timeout = int(self.config["scraping"]["timeout"])

        project_root = Path(config_path).resolve().parent.parent
        cache_dir = Path(self.config["scraping"]["cache_dir"])
        if not cache_dir.is_absolute():
            cache_dir = project_root / cache_dir
        self.cache_dir = cache_dir / "sherdog"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Referer": BASE_URL,
            }
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        value = str(name).strip().lower()
        value = re.sub(r"['.`]", "", value)
        value = value.replace("-", " ")
        value = re.sub(r"\s+", " ", value)
        return value

    def search_fighter(self, fighter_name: str) -> Optional[Dict]:
        """Search Sherdog Fight Finder for a fighter profile."""
        results = self._search_fighter_results(fighter_name)
        if not results:
            return None

        normalized_query = self._normalize_name(fighter_name)
        exact = [row for row in results if self._normalize_name(row.get("name", "")) == normalized_query]
        if exact:
            return exact[0]

        startswith = [row for row in results if self._normalize_name(row.get("name", "")).startswith(normalized_query)]
        if startswith:
            return startswith[0]

        contains = [row for row in results if normalized_query in self._normalize_name(row.get("name", ""))]
        if contains:
            return contains[0]

        return None

    def _search_fighter_results(self, fighter_name: str) -> List[Dict]:
        query = str(fighter_name).strip()
        if not query:
            return []

        search_url = f"{BASE_URL}/stats/fightfinder?SearchTxt={quote_plus(query)}"
        html_content = self._get(search_url)
        return self.parse_search_results(html_content)

    def scrape_fighter(
        self,
        fighter_url: str,
        fighter_id: Optional[str] = None,
        bust_cache: bool = False,
    ) -> Optional[Dict]:
        """Fetch and parse a Sherdog fighter page."""
        fighter_id = fighter_id or self._extract_fighter_id(fighter_url)
        cache_file = self.cache_dir / f"{fighter_id}.html"

        if bust_cache and cache_file.exists():
            cache_file.unlink()

        if cache_file.exists() and self.config["scraping"]["cache_enabled"]:
            html_content = cache_file.read_text(encoding="utf-8")
        else:
            try:
                response = self.session.get(fighter_url, timeout=self.timeout)
                response.raise_for_status()
                html_content = response.text
                cache_file.write_text(html_content, encoding="utf-8")
                time.sleep(self.rate_limit)
            except Exception as exc:
                logger.error(f"Error fetching Sherdog fighter page {fighter_url}: {exc}")
                return None

        return self.parse_fighter_html(html_content, fighter_url=fighter_url, fighter_id=fighter_id)

    def parse_fighter_html(self, html_content: str, *, fighter_url: str, fighter_id: Optional[str] = None) -> Dict:
        """Parse a Sherdog fighter page from raw HTML."""
        soup = BeautifulSoup(html_content, "lxml")
        resolved_id = fighter_id or self._extract_fighter_id(fighter_url)

        fighter_data = {
            "fighter_id": resolved_id,
            "source": "sherdog",
            "url": fighter_url,
            "scraped_at": datetime.now().isoformat(),
        }
        fighter_data.update(self._extract_basic_info(soup))
        fighter_data["method_breakdown"] = self._extract_method_breakdown(soup)
        fighter_data["fight_history"] = self._extract_fight_history(soup)
        return fighter_data

    def parse_search_results(self, html_content: str) -> List[Dict]:
        """Parse Sherdog Fight Finder fighter-result rows."""
        soup = BeautifulSoup(html_content, "lxml")
        table = None
        for header in soup.find_all(["h1", "h2", "h3", "h4"]):
            if "FIGHTER RESULTS" in header.get_text(" ", strip=True).upper():
                table = header.find_next("table", class_="fightfinder_result")
                break
        if table is None:
            table = soup.select_one("table.new_table.fightfinder_result")
        if table is None:
            return []

        results: List[Dict] = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            fighter_link = cols[1].find("a", href=re.compile(r"^/fighter/"))
            if fighter_link is None:
                continue
            fighter_url = urljoin(BASE_URL, fighter_link.get("href", ""))
            results.append(
                {
                    "name": fighter_link.get_text(" ", strip=True),
                    "url": fighter_url,
                    "fighter_id": self._extract_fighter_id(fighter_url),
                    "nickname": cols[2].get_text(" ", strip=True) or None,
                    "height": cols[3].get_text(" ", strip=True) or None,
                    "weight": cols[4].get_text(" ", strip=True) or None,
                    "association": cols[5].get_text(" ", strip=True) or None,
                }
            )
        return results

    def _extract_basic_info(self, soup: BeautifulSoup) -> Dict:
        info: Dict[str, object] = {}

        name_meta = soup.select_one('meta[itemprop="name"]')
        if name_meta and name_meta.get("content"):
            info["name"] = name_meta["content"].strip()

        nationality = soup.select_one('.fighter-nationality strong[itemprop="nationality"]')
        if nationality:
            info["nationality"] = nationality.get_text(" ", strip=True)

        birthplace = soup.select_one('.fighter-nationality .locality')
        if birthplace:
            info["birthplace"] = birthplace.get_text(" ", strip=True)

        association_link = soup.select_one(".association-class a.association")
        if association_link:
            info["association"] = association_link.get_text(" ", strip=True)
            info["association_url"] = urljoin(BASE_URL, association_link.get("href", ""))

        info_table = soup.select_one(".fighter-data .bio-holder table")
        if info_table:
            for row in info_table.select("tr"):
                cols = row.find_all("td")
                if len(cols) < 2:
                    continue
                label = cols[0].get_text(" ", strip=True).upper()
                value_text = cols[1].get_text(" ", strip=True)

                if label == "AGE":
                    info["age"] = self._parse_int(cols[1].find("b").get_text(strip=True) if cols[1].find("b") else value_text)
                    birth_date = cols[1].find(attrs={"itemprop": "birthDate"})
                    if birth_date:
                        info["date_of_birth"] = birth_date.get_text(" ", strip=True)
                elif label == "HEIGHT":
                    info["height"] = value_text
                    info["height_cm"] = self._parse_metric_value(value_text, unit="cm")
                elif label == "WEIGHT":
                    info["weight"] = value_text
                    info["weight_lbs"] = self._parse_int(cols[1].find("b").get_text(strip=True) if cols[1].find("b") else value_text)

        for line_break_parent in soup.select(".association-class"):
            text_lines = [line.strip() for line in line_break_parent.get_text("\n", strip=True).splitlines() if line.strip()]
            for idx, line in enumerate(text_lines):
                if line == "CLASS" and idx + 1 < len(text_lines):
                    info["weight_class"] = text_lines[idx + 1]
                    break

        return info

    def _extract_method_breakdown(self, soup: BeautifulSoup) -> Dict:
        breakdown: Dict[str, Dict[str, object]] = {}
        holder = soup.select_one(".winsloses-holder")
        if not holder:
            return breakdown

        for section in holder.find_all("div", recursive=False):
            total_node = section.select_one(".winloses")
            if not total_node:
                continue

            spans = total_node.find_all("span")
            if len(spans) < 2:
                continue

            section_name = spans[0].get_text(" ", strip=True).lower()
            section_data: Dict[str, object] = {"total": self._parse_int(spans[1].get_text(" ", strip=True)) or 0}

            titles = section.select(".meter-title")
            meters = section.select(".meter")
            for title_node, meter_node in zip(titles, meters):
                key = self._normalize_breakdown_key(title_node.get_text(" ", strip=True))
                parts = meter_node.find_all("div", recursive=False)
                count = self._parse_int(parts[0].get_text(" ", strip=True)) if parts else None
                pct_node = meter_node.select_one(".pr")
                percentage = self._parse_percentage(pct_node.get_text(" ", strip=True)) if pct_node else None
                section_data[key] = {"count": count, "percentage": percentage}

            breakdown[section_name] = section_data

        return breakdown

    def _extract_fight_history(self, soup: BeautifulSoup) -> List[Dict]:
        fights: List[Dict] = []
        rows = soup.select(".module.fight_history table.new_table.fighter tr")
        if not rows:
            return fights

        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue

            opponent_link = cols[1].find("a")
            event_link = cols[2].find("a")
            method_bold = cols[3].find("b")
            sub_lines = cols[3].select(".sub_line")
            referee_link = cols[3].find("a")

            fights.append(
                {
                    "result": cols[0].get_text(" ", strip=True).lower() or None,
                    "opponent": opponent_link.get_text(" ", strip=True) if opponent_link else cols[1].get_text(" ", strip=True) or None,
                    "opponent_url": urljoin(BASE_URL, opponent_link.get("href", "")) if opponent_link and opponent_link.get("href") else None,
                    "event": event_link.get_text(" ", strip=True) if event_link else None,
                    "event_url": urljoin(BASE_URL, event_link.get("href", "")) if event_link and event_link.get("href") else None,
                    "date": self._normalize_sherdog_date(cols[2].select_one(".sub_line").get_text(" ", strip=True) if cols[2].select_one(".sub_line") else None),
                    "method": method_bold.get_text(" ", strip=True) if method_bold else cols[3].get_text(" ", strip=True) or None,
                    "referee": referee_link.get_text(" ", strip=True) if referee_link else (sub_lines[0].get_text(" ", strip=True) if sub_lines else None),
                    "round": self._parse_int(cols[4].get_text(" ", strip=True)),
                    "time": cols[5].get_text(" ", strip=True) or None,
                }
            )

        return fights

    def _get(self, url: str) -> str:
        cache_key = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_") + ".html"
        cache_file = self.cache_dir / cache_key

        if cache_file.exists() and self.config["scraping"]["cache_enabled"]:
            return cache_file.read_text(encoding="utf-8")

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        html_content = response.text
        cache_file.write_text(html_content, encoding="utf-8")
        time.sleep(self.rate_limit)
        return html_content

    @staticmethod
    def _extract_fighter_id(fighter_url: str) -> str:
        fighter_url = fighter_url.rstrip("/")
        match = re.search(r"-(\d+)$", fighter_url)
        if match:
            return match.group(1)
        return re.sub(r"[^a-zA-Z0-9]+", "_", fighter_url).strip("_")

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        match = re.search(r"(\d+)", value)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_percentage(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)%", value)
        return float(match.group(1)) / 100.0 if match else None

    @staticmethod
    def _parse_metric_value(value: Optional[str], *, unit: str) -> Optional[float]:
        if not value:
            return None
        match = re.search(rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}", value, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    @staticmethod
    def _normalize_breakdown_key(label: str) -> str:
        cleaned = label.lower().replace("/", " ")
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
        return cleaned.strip("_")

    @staticmethod
    def _normalize_sherdog_date(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = re.sub(r"\s*/\s*", " ", value.strip())
        try:
            return datetime.strptime(cleaned, "%b %d %Y").strftime("%Y-%m-%d")
        except ValueError:
            return value


def main():
    parser = argparse.ArgumentParser(description="Scrape a Sherdog fighter page")
    parser.add_argument("--url", required=True, help="Full Sherdog fighter URL")
    parser.add_argument("--fighter-id", help="Override extracted Sherdog fighter ID")
    parser.add_argument("--bust-cache", action="store_true", help="Refetch even if cached")
    parser.add_argument("--output", help="Write JSON output to this path")
    args = parser.parse_args()

    scraper = SherdogScraper()
    fighter = scraper.scrape_fighter(args.url, fighter_id=args.fighter_id, bust_cache=args.bust_cache)
    if fighter is None:
        raise SystemExit(1)

    payload = json.dumps(fighter, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        logger.success(f"Wrote Sherdog fighter data to {output_path}")
        return

    print(payload)


if __name__ == "__main__":
    main()
