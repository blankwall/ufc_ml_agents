"""
Event Scraper - Extracts event and fight data from UFCStats.com
"""

import time
import re
from typing import Dict, List, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from loguru import logger
import yaml
from datetime import datetime


class EventScraper:
    """Scrapes UFC event pages to extract fight cards and results"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the event scraper with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['data_sources']['ufcstats']['base_url']
        self.events_url = self.config['data_sources']['ufcstats']['events_url']
        self.rate_limit = self.config['data_sources']['ufcstats']['rate_limit']
        self.user_agent = self.config['scraping']['user_agent']
        self.timeout = self.config['scraping']['timeout']
        self.cache_dir = Path(self.config['scraping']['cache_dir']) / "events"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def get_all_event_links(self, completed_only: bool = True, max_pages: int = None) -> List[Dict[str, str]]:
        """
        Scrape the events list page to get all event URLs
        
        Args:
            completed_only: If True, only return completed events with results
            max_pages: Maximum number of pages to scrape (None = all pages)
            
        Returns:
            List of dicts with event information
        """
        logger.info("Fetching event list...")
        events = []
        
        base_url = self.events_url if completed_only else f"{self.base_url}/statistics/events/upcoming"
        
        # First, get the first page to determine total number of pages
        try:
            response = self.session.get(base_url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Find pagination links to determine total pages
            pagination_links = soup.select('a.b-statistics__paginate-link')
            max_page = 1
            
            for link in pagination_links:
                page_text = link.text.strip()
                if page_text.isdigit():
                    max_page = max(max_page, int(page_text))
            
            # Also check for "All" link which might indicate total pages
            if not pagination_links:
                # Try to find the last page number in the pagination
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    if 'page=' in link['href']:
                        try:
                            page_num = int(link['href'].split('page=')[-1])
                            max_page = max(max_page, page_num)
                        except:
                            pass
            
            # Limit pages if max_pages is specified
            if max_pages:
                max_page = min(max_page, max_pages)
                logger.info(f"Found {max_page} pages of events to scrape (limited to {max_pages})")
            else:
                logger.info(f"Found {max_page} pages of events to scrape")
            
        except Exception as e:
            logger.error(f"Error determining page count: {e}")
            max_page = 1
        
        # Now iterate through all pages
        for page in range(1, max_page + 1):
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?page={page}"
            
            logger.info(f"Fetching page {page}/{max_page}...")
            
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'lxml')
                event_rows = soup.select('tr.b-statistics__table-row')
                
                page_events = 0
                for row in event_rows[1:]:  # Skip header row
                    link = row.select_one('a.b-link.b-link_style_black')
                    if link:
                        event_name = link.text.strip()
                        event_url = link['href']
                        
                        # Extract event ID from URL
                        if event_url and '/' in event_url:
                            event_id = event_url.rstrip('/').split('/')[-1]
                        else:
                            continue
                        
                        # Extract date and location from the text in the link
                        # The date is usually after the event name
                        full_text = link.text.strip()
                        # Split by multiple spaces or newlines to separate name and date
                        parts = [p.strip() for p in full_text.split('\n') if p.strip()]
                        
                        event_name = parts[0] if parts else event_name
                        event_date = parts[1] if len(parts) > 1 else None
                        
                        # Location is in a separate column
                        cols = row.select('td.b-statistics__table-col')
                        location = None
                        if len(cols) > 1:
                            location = cols[1].text.strip()
                        
                        events.append({
                            'name': event_name,
                            'url': event_url,
                            'event_id': event_id,
                            'date': event_date,
                            'location': location
                        })
                        page_events += 1
                
                logger.info(f"  Found {page_events} events on page {page}")
                
                # Rate limiting
                time.sleep(self.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                continue
        
        logger.success(f"Found total of {len(events)} events across {max_page} pages")
        return events
    
    def scrape_event(self, event_url: str, event_id: str) -> Optional[Dict]:
        """
        Scrape detailed information for a single event
        
        Args:
            event_url: URL of the event page
            event_id: Unique event identifier
            
        Returns:
            Dictionary containing event data and all fights
        """
        cache_file = self.cache_dir / f"{event_id}.html"
        
        # Check cache first
        if cache_file.exists() and self.config['scraping']['cache_enabled']:
            logger.debug(f"Loading event {event_id} from cache")
            with open(cache_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            try:
                logger.debug(f"Fetching event page: {event_url}")
                response = self.session.get(event_url, timeout=self.timeout)
                response.raise_for_status()
                html_content = response.text
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching event {event_id}: {e}")
                return None
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        try:
            event_data = {
                'event_id': event_id,
                'url': event_url,
                'scraped_at': datetime.now().isoformat(),
            }
            
            # Extract event info
            event_data.update(self._extract_event_info(soup))
            
            # Extract all fights
            event_data['fights'] = self._extract_fights(soup)
            
            logger.success(f"Successfully scraped event: {event_data.get('name', event_id)} with {len(event_data['fights'])} fights")
            return event_data
            
        except Exception as e:
            logger.error(f"Error parsing event {event_id}: {e}")
            return None
    
    def scrape_event_from_file(self, file_path: str) -> Optional[Dict]:
        """
        Parse an event from a saved HTML file
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Dictionary containing event data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract event ID from the HTML if possible
            event_id = Path(file_path).stem
            
            event_data = {
                'event_id': event_id,
                'source_file': file_path,
                'scraped_at': datetime.now().isoformat(),
            }
            
            event_data.update(self._extract_event_info(soup))
            event_data['fights'] = self._extract_fights(soup)
            
            logger.success(f"Parsed event from file: {event_data.get('name', event_id)}")
            return event_data
            
        except Exception as e:
            logger.error(f"Error parsing event file {file_path}: {e}")
            return None
    
    def _extract_event_info(self, soup: BeautifulSoup) -> Dict:
        """Extract event information"""
        info = {}
        
        # Event name
        title_elem = soup.select_one('span.b-content__title-highlight')
        info['name'] = title_elem.text.strip() if title_elem else None
        
        # Event details (date, location)
        info_items = soup.select('li.b-list__box-list-item')
        for item in info_items:
            title = item.select_one('i.b-list__box-item-title')
            if not title:
                continue
            
            title_text = title.text.strip().rstrip(':').lower()
            value_text = item.text.replace(title.text, '').strip()
            
            if title_text == 'date':
                info['date'] = value_text
            elif title_text == 'location':
                info['location'] = value_text
            elif title_text == 'enclosure':
                info['venue'] = value_text
        
        return info
    
    def _extract_fights(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all fights from the event"""
        fights = []
        
        # Find all fight rows
        fight_rows = soup.select('tr.b-fight-details__table-row')
        
        for i, row in enumerate(fight_rows[1:], 1):  # Skip header row
            try:
                fight = self._parse_fight_row(row, i)
                if fight:
                    fights.append(fight)
            except Exception as e:
                logger.warning(f"Error parsing fight row {i}: {e}")
                continue
        
        return fights
    
    def _parse_fight_row(self, row, fight_number: int) -> Optional[Dict]:
        """Parse a single fight row"""
        cols = row.select('td.b-fight-details__table-col')
        
        if len(cols) < 2:
            return None
        
        fight = {
            'fight_number': fight_number,
        }
        
        # Fighter names and links (column 1)
        fighter_links = cols[1].select('a.b-link.b-link_style_black')
        if len(fighter_links) >= 2:
            fight['fighter_1_name'] = fighter_links[0].text.strip()
            fight['fighter_1_url'] = fighter_links[0]['href']
            fight['fighter_1_id'] = fighter_links[0]['href'].split('/')[-1]
            
            fight['fighter_2_name'] = fighter_links[1].text.strip()
            fight['fighter_2_url'] = fighter_links[1]['href']
            fight['fighter_2_id'] = fighter_links[1]['href'].split('/')[-1]
        
        # Winner indication (red/blue corner wins shown in column 0)
        result_imgs = cols[0].select('i')
        if result_imgs:
            # Check which fighter won (top or bottom)
            fight['result'] = self._determine_winner(cols[0])
        
        # Weight class (column 6)
        if len(cols) > 6:
            weight_class = cols[6].text.strip()
            fight['weight_class'] = weight_class
            
            # Check for title fight
            if 'title' in weight_class.lower():
                fight['is_title_fight'] = True
            else:
                fight['is_title_fight'] = False
        
        # Method of victory (column 7)
        if len(cols) > 7:
            method_col = cols[7]
            method_parts = method_col.text.strip().split('\n')
            fight['method'] = method_parts[0].strip() if method_parts else None
            
            # Sometimes there's a detail (e.g., "KO/TKO - Punches")
            if len(method_parts) > 1:
                fight['method_detail'] = method_parts[1].strip()
        
        # Round (column 8)
        if len(cols) > 8:
            fight['round'] = self._parse_int(cols[8].text.strip())
        
        # Time (column 9)
        if len(cols) > 9:
            fight['time'] = cols[9].text.strip()
        
        # Fight detail link
        data_link = row.get('data-link')
        if data_link:
            fight['fight_detail_url'] = data_link
            fight['fight_detail_id'] = data_link.split('/')[-1]
        
        return fight
    
    def _determine_winner(self, result_col) -> Optional[str]:
        """Determine fight winner from result column"""
        # Look for win indicators in the HTML structure
        text = result_col.text.strip().lower()
        
        if 'win' in text:
            # The structure usually has the winner indicated
            # This needs to be refined based on actual HTML structure
            if result_col.select_one('i.b-flag__text'):
                return 'fighter_1'  # Top fighter won
            else:
                return 'fighter_2'  # Bottom fighter won
        
        # Check for draw or no contest
        if 'draw' in text:
            return 'draw'
        if 'nc' in text or 'no contest' in text:
            return 'no_contest'
        
        return None
    
    def _parse_int(self, value_str: str) -> Optional[int]:
        """Parse integer value"""
        if not value_str or value_str == '--':
            return None
        try:
            return int(value_str)
        except ValueError:
            return None
    
    def scrape_fight_details(self, fight_url: str) -> Optional[Dict]:
        """
        Scrape detailed round-by-round statistics for a specific fight
        
        Args:
            fight_url: URL of the fight details page
            
        Returns:
            Dictionary with detailed fight statistics
        """
        fight_id = fight_url.split('/')[-1]
        cache_file = self.cache_dir / f"fight_{fight_id}.html"
        
        # Check cache
        if cache_file.exists() and self.config['scraping']['cache_enabled']:
            with open(cache_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            try:
                response = self.session.get(fight_url, timeout=self.timeout)
                response.raise_for_status()
                html_content = response.text
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                time.sleep(self.rate_limit)
            except Exception as e:
                logger.error(f"Error fetching fight details {fight_id}: {e}")
                return None
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        try:
            # Extract detailed stats
            fight_stats = {
                'fight_id': fight_id,
                'url': fight_url,
            }
            
            # Extract fight outcome and details
            outcome = self._extract_fight_outcome(soup)
            fight_stats.update(outcome)
            
            # Extract fighter names and IDs (from outcome or fallback)
            if 'fighter_1_name' not in fight_stats:
                fighter_names = self._extract_fighter_names(soup)
                fight_stats['fighter_1_name'] = fighter_names.get('fighter_1')
                fight_stats['fighter_2_name'] = fighter_names.get('fighter_2')
                # Also get IDs if available
                if 'fighter_1_id' not in fight_stats:
                    fight_stats['fighter_1_id'] = fighter_names.get('fighter_1_id')
                if 'fighter_2_id' not in fight_stats:
                    fight_stats['fighter_2_id'] = fighter_names.get('fighter_2_id')
            
            # Round-by-round totals
            fight_stats['totals'] = self._extract_totals_table(soup)
            
            # Significant strikes breakdown
            fight_stats['significant_strikes'] = self._extract_significant_strikes_table(soup)
            
            logger.debug(f"Extracted detailed stats for fight {fight_id}")
            return fight_stats
            
        except Exception as e:
            logger.error(f"Error parsing fight details {fight_id}: {e}")
            return None
    
    def _extract_fight_outcome(self, soup: BeautifulSoup) -> Dict:
        """Extract fight outcome, method, round, time, etc."""
        outcome = {}
        
        try:
            # Find the fight details section at the top
            # Look for W/L indicators and fighter names
            fight_sections = soup.select('div.b-fight-details__person')
            
            if len(fight_sections) >= 2:
                # Fighter 1 (left side)
                f1_section = fight_sections[0]
                f1_result = f1_section.select_one('i.b-fight-details__person-status')
                f1_link = f1_section.select_one('a.b-link.b-link_style_black')
                
                if f1_link:
                    outcome['fighter_1_name'] = f1_link.text.strip()
                    # Extract fighter ID from the link URL
                    f1_href = f1_link.get('href', '')
                    if f1_href and '/fighter-details/' in f1_href:
                        f1_id = f1_href.split('/fighter-details/')[-1].rstrip('/')
                        outcome['fighter_1_id'] = f1_id
                if f1_result:
                    f1_result_text = f1_result.text.strip()
                    outcome['fighter_1_result'] = f1_result_text  # 'W', 'L', or 'D'
                
                # Fighter 2 (right side)
                f2_section = fight_sections[1]
                f2_result = f2_section.select_one('i.b-fight-details__person-status')
                f2_link = f2_section.select_one('a.b-link.b-link_style_black')
                
                if f2_link:
                    outcome['fighter_2_name'] = f2_link.text.strip()
                    # Extract fighter ID from the link URL
                    f2_href = f2_link.get('href', '')
                    if f2_href and '/fighter-details/' in f2_href:
                        f2_id = f2_href.split('/fighter-details/')[-1].rstrip('/')
                        outcome['fighter_2_id'] = f2_id
                if f2_result:
                    f2_result_text = f2_result.text.strip()
                    outcome['fighter_2_result'] = f2_result_text
                
                # Determine winner
                if outcome.get('fighter_1_result') == 'W':
                    outcome['winner'] = 'fighter_1'
                elif outcome.get('fighter_2_result') == 'W':
                    outcome['winner'] = 'fighter_2'
                elif outcome.get('fighter_1_result') == 'D':
                    outcome['winner'] = 'draw'
                else:
                    outcome['winner'] = None
            
            # Extract method, round, time from the fight details section
            fight_details_text = soup.select('p.b-fight-details__text')
            
            for text_section in fight_details_text:
                # Find all text items in this section
                text_items = text_section.find_all('i')
                
                for item in text_items:
                    # Get the label
                    label = item.select_one('i.b-fight-details__label')
                    if not label:
                        continue
                    
                    label_text = label.text.strip().lower()
                    
                    if 'method:' in label_text:
                        # The method value is in a nested <i> with style="font-style: normal"
                        method_value = item.find('i', style=re.compile('font-style'))
                        if method_value:
                            outcome['method'] = method_value.text.strip()
                    
                    elif 'round:' in label_text:
                        # Get text content after removing the label
                        # The round number is directly after the label
                        full_text = item.get_text(separator=' ', strip=True)
                        round_text = full_text.replace('Round:', '').strip()
                        try:
                            outcome['round'] = int(round_text)
                        except ValueError:
                            # If it's not a number, store as string
                            outcome['round'] = round_text
                    
                    elif 'time:' in label_text and 'format' not in label_text:
                        # Get just the text content, not the label
                        full_text = item.get_text(separator=' ', strip=True)
                        # Remove the label
                        time_text = full_text.replace('Time:', '').strip()
                        outcome['time'] = time_text
                    
                    elif 'time format:' in label_text:
                        # Get just the text content
                        full_text = item.get_text(separator=' ', strip=True)
                        format_text = full_text.replace('Time format:', '').strip()
                        outcome['time_format'] = format_text
                    
                    elif 'referee:' in label_text:
                        # Referee name might be in a span
                        referee_span = item.find('span')
                        if referee_span:
                            outcome['referee'] = referee_span.text.strip()
                        else:
                            full_text = item.text.strip()
                            referee = full_text.replace(label.text, '').strip()
                            outcome['referee'] = referee
                    
                    elif 'details:' in label_text:
                        # Details come after this item
                        full_text = item.parent.text.strip()
                        details = full_text.replace('Details:', '').strip()
                        if details:
                            outcome['method_details'] = details
            
        except Exception as e:
            logger.warning(f"Error extracting fight outcome: {e}")
        
        return outcome
    
    def _extract_fighter_names(self, soup: BeautifulSoup) -> Dict:
        """Extract fighter names and IDs from fight detail page (fallback method)"""
        names = {}
        
        # Fighter names are typically in specific elements
        fighter_links = soup.select('a.b-link.b-link_style_black')
        
        if len(fighter_links) >= 2:
            names['fighter_1'] = fighter_links[0].text.strip()
            # Extract fighter ID from link
            f1_href = fighter_links[0].get('href', '')
            if f1_href and '/fighter-details/' in f1_href:
                names['fighter_1_id'] = f1_href.split('/fighter-details/')[-1].rstrip('/')
            
            names['fighter_2'] = fighter_links[1].text.strip()
            # Extract fighter ID from link
            f2_href = fighter_links[1].get('href', '')
            if f2_href and '/fighter-details/' in f2_href:
                names['fighter_2_id'] = f2_href.split('/fighter-details/')[-1].rstrip('/')
        
        return names
    
    def _extract_totals_table(self, soup: BeautifulSoup) -> Dict:
        """Extract the totals statistics table"""
        totals = {
            'fighter_1': {},
            'fighter_2': {}
        }
        
        try:
            # Find ALL tables in the fight details section
            # The totals table may or may not have specific classes
            tables = soup.find_all('table')
            
            for table in tables:
                # Check if this has a thead with the right headers
                thead = table.find('thead', class_='b-fight-details__table-head')
                if not thead:
                    continue
                
                # Check if it's the round-by-round table (skip that)
                if 'b-fight-details__table-head_rnd' in str(thead.get('class', [])):
                    continue
                
                # Get headers
                headers = []
                for th in thead.find_all('th'):
                    headers.append(th.text.strip().lower())
                
                # Check if this looks like the totals table (has 'kd', 'sig. str.' etc)
                has_totals_cols = any(h in headers for h in ['kd', 'sig. str.', 'td'])
                if not has_totals_cols:
                    continue
                
                # Get data rows
                tbody = table.find('tbody', class_='b-fight-details__table-body')
                if not tbody:
                    continue
                
                rows = tbody.find_all('tr', class_='b-fight-details__table-row')
                if not rows:
                    continue
                
                # Process first data row
                data_row = rows[0]
                cols = data_row.find_all('td', class_='b-fight-details__table-col')
                
                # Map stat names
                stat_mapping = {
                    'kd': 'knockdowns',
                    'sig. str.': 'sig_strikes',
                    'sig. str. %': 'sig_strike_pct',
                    'total str.': 'total_strikes',
                    'td': 'takedowns',
                    'td %': 'takedown_pct',
                    'sub. att': 'submission_attempts',
                    'rev.': 'reversals',
                    'ctrl': 'control_time'
                }
                
                # Each column has TWO <p> tags - one for each fighter
                for i, col in enumerate(cols):
                    if i >= len(headers):
                        continue
                    
                    header = headers[i]
                    
                    # Get the two <p> tags in this column
                    p_tags = col.find_all('p', class_='b-fight-details__table-text')
                    
                    if header in stat_mapping:
                        stat_key = stat_mapping[header]
                        
                        if len(p_tags) >= 2:
                            totals['fighter_1'][stat_key] = p_tags[0].text.strip()
                            totals['fighter_2'][stat_key] = p_tags[1].text.strip()
                        elif len(p_tags) == 1:
                            # Single value for both
                            totals['fighter_1'][stat_key] = p_tags[0].text.strip()
                            totals['fighter_2'][stat_key] = p_tags[0].text.strip()
                
                break  # Found and processed totals
            
        except Exception as e:
            logger.warning(f"Error extracting totals: {e}")
        
        return totals
    
    def _extract_significant_strikes_table(self, soup: BeautifulSoup) -> Dict:
        """Extract significant strikes breakdown table"""
        sig_strikes = {
            'fighter_1': {},
            'fighter_2': {}
        }
        
        try:
            # Find all tables
            tables = soup.find_all('table')
            
            # The significant strikes table is typically the second table
            # Or we can identify it by checking if it has different headers
            for table in tables:
                thead = table.find('thead')
                if not thead:
                    continue
                
                # Get headers to identify the table
                headers = []
                for th in thead.find_all('th'):
                    header_text = th.text.strip().lower()
                    headers.append(header_text)
                
                # Check if this is the sig strikes table (has 'head', 'body', 'leg' columns)
                has_sig_strike_cols = any(h in headers for h in ['head', 'body', 'leg'])
                
                if not has_sig_strike_cols:
                    continue
                
                # Found significant strikes table
                tbody = table.find('tbody')
                if not tbody:
                    continue
                
                rows = tbody.find_all('tr')
                if not rows:
                    continue
                
                # Process first data row
                data_row = rows[0]
                cols = data_row.find_all('td')
                
                # Stat mapping
                stat_mapping = {
                    'sig. str': 'sig_strikes_total',
                    'sig. str.': 'sig_strikes_total',
                    'sig. str. %': 'sig_strike_pct',
                    'head': 'head_strikes',
                    'body': 'body_strikes',
                    'leg': 'leg_strikes',
                    'distance': 'distance_strikes',
                    'clinch': 'clinch_strikes',
                    'ground': 'ground_strikes'
                }
                
                # Each column has TWO <p> tags - one for each fighter
                for i, col in enumerate(cols):
                    if i >= len(headers):
                        continue
                    
                    header = headers[i]
                    
                    # Get the two <p> tags in this column
                    p_tags = col.find_all('p', class_='b-fight-details__table-text')
                    
                    if header in stat_mapping:
                        stat_key = stat_mapping[header]
                        
                        if len(p_tags) >= 2:
                            sig_strikes['fighter_1'][stat_key] = p_tags[0].text.strip()
                            sig_strikes['fighter_2'][stat_key] = p_tags[1].text.strip()
                        elif len(p_tags) == 1:
                            sig_strikes['fighter_1'][stat_key] = p_tags[0].text.strip()
                            sig_strikes['fighter_2'][stat_key] = p_tags[0].text.strip()
                
                break  # Found and processed
            
        except Exception as e:
            logger.warning(f"Error extracting significant strikes: {e}")
        
        return sig_strikes
    
    def scrape_all_fight_details(self, events_file: str, output_file: str = None):
        """
        Scrape detailed stats for all fights from events file
        
        Args:
            events_file: Path to events JSON file
            output_file: Path to save fight details (optional)
        """
        import json
        
        logger.info(f"Loading events from {events_file}")
        
        with open(events_file, 'r') as f:
            events = json.load(f)
        
        all_fight_details = []
        total_fights = sum(len(event.get('fights', [])) for event in events)
        
        logger.info(f"Found {total_fights} fights across {len(events)} events")
        
        fight_count = 0
        for event in events:
            event_name = event.get('name', 'Unknown')
            logger.info(f"\nProcessing event: {event_name}")
            
            for fight in event.get('fights', []):
                fight_count += 1
                fight_detail_url = fight.get('fight_detail_url')
                
                if not fight_detail_url:
                    logger.debug(f"  No fight detail URL for fight {fight_count}")
                    continue
                
                logger.info(f"  [{fight_count}/{total_fights}] Scraping fight: {fight.get('fighter_1_name')} vs {fight.get('fighter_2_name')}")
                
                # Scrape fight details
                details = self.scrape_fight_details(fight_detail_url)
                
                if details:
                    # Add event and fight context
                    details['event_id'] = event.get('event_id')
                    details['event_name'] = event_name
                    details['fighter_1_id'] = fight.get('fighter_1_id')
                    details['fighter_2_id'] = fight.get('fighter_2_id')
                    
                    all_fight_details.append(details)
                
                # Save progress every 50 fights
                if output_file and fight_count % 50 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(all_fight_details, f, indent=2)
                    logger.info(f"  Progress saved: {len(all_fight_details)} fight details")
        
        # Final save
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_fight_details, f, indent=2)
            logger.success(f"Saved {len(all_fight_details)} fight details to {output_file}")
        
        return all_fight_details


def main():
    """Main function for running the scraper from command line"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Scrape UFC event data')
    parser.add_argument('--mode', choices=['all', 'single', 'file', 'test', 'fight-details'], default='all',
                       help='Scrape all events, single event, parse from file, test with 3 pages, or scrape fight details')
    parser.add_argument('--event-id', type=str,
                       help='Event ID for single event mode')
    parser.add_argument('--file', type=str,
                       help='Path to HTML file to parse')
    parser.add_argument('--events-file', type=str,
                       help='Path to events JSON file (for fight-details mode)')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum number of pages to scrape (for testing)')
    parser.add_argument('--output', type=str, default='data/processed/events.json',
                       help='Output file for scraped data')
    
    args = parser.parse_args()
    
    scraper = EventScraper()
    
    if args.mode == 'all' or args.mode == 'test':
        # Determine max pages
        if args.mode == 'test':
            max_pages = 3  # Test with just 3 pages
        else:
            max_pages = args.max_pages  # None = all pages
        
        # Get all event links
        event_links = scraper.get_all_event_links(max_pages=max_pages)
        
        # Scrape each event
        all_events = []
        for i, event_info in enumerate(event_links, 1):
            logger.info(f"Scraping event {i}/{len(event_links)}: {event_info['name']}")
            event_data = scraper.scrape_event(event_info['url'], event_info['event_id'])
            if event_data:
                all_events.append(event_data)
            
            # Save progress every 10 events
            if i % 10 == 0:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(all_events, f, indent=2)
                logger.info(f"Progress saved: {len(all_events)} events")
        
        # Final save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_events, f, indent=2)
        
        logger.success(f"Scraped {len(all_events)} events. Saved to {args.output}")
    
    elif args.mode == 'single':
        if not args.event_id:
            logger.error("--event-id required for single event mode")
            return
        
        event_url = f"{scraper.base_url}/event-details/{args.event_id}"
        event_data = scraper.scrape_event(event_url, args.event_id)
        
        if event_data:
            print(json.dumps(event_data, indent=2))
    
    elif args.mode == 'file':
        if not args.file:
            logger.error("--file required for file mode")
            return
        
        event_data = scraper.scrape_event_from_file(args.file)
        
        if event_data:
            print(json.dumps(event_data, indent=2))
    
    elif args.mode == 'fight-details':
        if not args.events_file:
            logger.error("--events-file required for fight-details mode")
            logger.info("Example: python scrapers/event_scraper.py --mode fight-details --events-file data/processed/events.json --output data/processed/fight_details.json")
            return
        
        output_file = args.output if args.output != 'data/processed/events.json' else 'data/processed/fight_details.json'
        scraper.scrape_all_fight_details(args.events_file, output_file)


if __name__ == '__main__':
    main()

