"""
UFC Data Scraping Module

Provides scrapers for collecting fighter stats, event data, and fight history
from UFCStats.com.
"""

from .fighter_scraper import FighterScraper
from .event_scraper import EventScraper

__all__ = ['FighterScraper', 'EventScraper']

