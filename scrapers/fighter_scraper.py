"""
Fighter Scraper - Extracts fighter statistics and fight history from UFCStats.com
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


class FighterScraper:
    """Scrapes individual fighter pages from UFCStats.com"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the fighter scraper with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['data_sources']['ufcstats']['base_url']
        self.rate_limit = self.config['data_sources']['ufcstats']['rate_limit']
        self.user_agent = self.config['scraping']['user_agent']
        self.timeout = self.config['scraping']['timeout']
        self.cache_dir = Path(self.config['scraping']['cache_dir']) / "fighters"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def get_all_fighter_links(self, test_mode: bool = False) -> List[Dict[str, str]]:
        """
        Scrape the fighter list page to get all fighter URLs
        
        Args:
            test_mode: If True, only scrape fighters starting with 'a' for testing
        
        Returns:
            List of dicts with 'name' and 'url' for each fighter
        """
        logger.info("Fetching fighter list...")
        fighters = []
        
        # The fighter list is paginated by letter
        letters = ['a'] if test_mode else list('abcdefghijklmnopqrstuvwxyz')
        
        for letter in letters:
            url = f"{self.base_url}/statistics/fighters?char={letter}&page=all"
            logger.debug(f"Fetching fighters starting with '{letter.upper()}'")
            
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'lxml')
                fighter_rows = soup.select('tr.b-statistics__table-row')
                
                for row in fighter_rows[1:]:  # Skip header row
                    links = row.select('a.b-link.b-link_style_black')
                    if len(links) >= 2:  # First and Last name links
                        # First name link
                        first_name = links[0].text.strip()
                        # Last name link  
                        last_name = links[1].text.strip()
                        fighter_name = f"{first_name} {last_name}"
                        
                        # Use the first link's URL (both should point to same fighter)
                        fighter_url = links[0]['href']
                        
                        # Extract fighter ID from URL
                        if fighter_url and '/' in fighter_url:
                            fighter_id = fighter_url.rstrip('/').split('/')[-1]
                            
                            if fighter_id:  # Make sure we got a valid ID
                                fighters.append({
                                    'name': fighter_name,
                                    'url': fighter_url,
                                    'fighter_id': fighter_id
                                })
                
                logger.info(f"Found {len([f for f in fighters if f['name'][0].lower() == letter])} fighters for '{letter.upper()}'")
                time.sleep(self.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching fighters for letter {letter}: {e}")
                continue
        
        logger.success(f"Found total of {len(fighters)} fighters")
        return fighters
    
    def scrape_fighter(self, fighter_url: str, fighter_id: str) -> Optional[Dict]:
        """
        Scrape detailed statistics for a single fighter
        
        Args:
            fighter_url: URL of the fighter's page
            fighter_id: Unique fighter identifier
            
        Returns:
            Dictionary containing all fighter data
        """
        cache_file = self.cache_dir / f"{fighter_id}.html"
        
        # Check cache first
        if cache_file.exists() and self.config['scraping']['cache_enabled']:
            logger.debug(f"Loading fighter {fighter_id} from cache")
            with open(cache_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            try:
                logger.debug(f"Fetching fighter page: {fighter_url}")
                response = self.session.get(fighter_url, timeout=self.timeout)
                response.raise_for_status()
                html_content = response.text
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching fighter {fighter_id}: {e}")
                return None
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        try:
            fighter_data = {
                'fighter_id': fighter_id,
                'url': fighter_url,
                'scraped_at': datetime.now().isoformat(),
            }
            
            # Extract basic info
            fighter_data.update(self._extract_basic_info(soup))
            
            # Extract career statistics
            fighter_data.update(self._extract_career_stats(soup))
            
            # Extract fight history
            fighter_data['fight_history'] = self._extract_fight_history(soup)
            
            logger.success(f"Successfully scraped fighter: {fighter_data.get('name', fighter_id)}")
            return fighter_data
            
        except Exception as e:
            logger.error(f"Error parsing fighter {fighter_id}: {e}")
            return None
    
    def _extract_basic_info(self, soup: BeautifulSoup) -> Dict:
        """Extract basic fighter information"""
        info = {}
        
        # Name
        name_elem = soup.select_one('span.b-content__title-highlight')
        info['name'] = name_elem.text.strip() if name_elem else None
        
        # Record (Wins-Losses-Draws)
        record_elem = soup.select_one('span.b-content__title-record')
        if record_elem:
            record_text = record_elem.text.strip()
            match = re.search(r'Record:\s*(\d+)-(\d+)-(\d+)', record_text)
            if match:
                info['wins'] = int(match.group(1))
                info['losses'] = int(match.group(2))
                info['draws'] = int(match.group(3))
        
        # Nickname
        nickname_elem = soup.select_one('p.b-content__Nickname')
        info['nickname'] = nickname_elem.text.strip() if nickname_elem else None
        
        # Physical attributes
        info_items = soup.select('li.b-list__box-list-item')
        for item in info_items:
            title = item.select_one('i.b-list__box-item-title')
            if not title:
                continue
            
            title_text = title.text.strip().rstrip(':').lower()
            value_text = item.text.replace(title.text, '').strip()
            
            if title_text == 'height':
                info['height'] = value_text
                info['height_cm'] = self._parse_height(value_text)
            elif title_text == 'weight':
                info['weight'] = value_text
                info['weight_lbs'] = self._parse_weight(value_text)
            elif title_text == 'reach':
                info['reach'] = value_text
                info['reach_inches'] = self._parse_reach(value_text)
            elif title_text == 'stance':
                info['stance'] = value_text
            elif title_text == 'dob':
                info['date_of_birth'] = value_text
                info['age'] = self._calculate_age(value_text)
        
        return info
    
    def _extract_career_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract career statistics"""
        stats = {}
        
        # Find all stat items
        stat_items = soup.select('li.b-list__box-list-item')
        
        for item in stat_items:
            title = item.select_one('i.b-list__box-item-title')
            if not title:
                continue
            
            title_text = title.text.strip().rstrip(':').lower()
            value_text = item.text.replace(title.text, '').strip()
            
            # Map stat names to keys
            stat_mapping = {
                'slpm': 'sig_strikes_landed_per_min',
                'str. acc.': 'striking_accuracy',
                'sapm': 'sig_strikes_absorbed_per_min',
                'str. def': 'striking_defense',
                'td avg.': 'takedown_avg_per_15min',
                'td acc.': 'takedown_accuracy',
                'td def.': 'takedown_defense',
                'sub. avg.': 'submission_avg_per_15min',
            }
            
            if title_text in stat_mapping:
                stats[stat_mapping[title_text]] = self._parse_stat_value(value_text)
        
        return stats
    
    def _extract_fight_history(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract complete fight history"""
        fights = []
        
        fight_rows = soup.select('tr.b-fight-details__table-row')
        
        for row in fight_rows[1:]:  # Skip header
            try:
                cols = row.select('td.b-fight-details__table-col')
                if len(cols) < 9:
                    continue
                
                fight = {
                    'result': cols[0].text.strip(),
                    'opponent': cols[1].select_one('a').text.strip() if cols[1].select_one('a') else None,
                    'opponent_url': cols[1].select_one('a')['href'] if cols[1].select_one('a') else None,
                    'knockout': cols[2].text.strip(),
                    'submission': cols[3].text.strip(),
                    'method': cols[7].text.strip(),
                    'round': self._parse_int(cols[8].text.strip()),
                    'time': cols[9].text.strip(),
                    'event': cols[6].select_one('a').text.strip() if cols[6].select_one('a') else None,
                    'event_url': cols[6].select_one('a')['href'] if cols[6].select_one('a') else None,
                    'date': cols[6].select_one('p.b-fight-details__table-text').text.strip() if cols[6].select_one('p.b-fight-details__table-text') else None,
                }
                
                # Extract fight detail stats if available
                fight_detail_link = row.get('data-link')
                if fight_detail_link:
                    fight['fight_detail_url'] = fight_detail_link
                
                fights.append(fight)
                
            except Exception as e:
                logger.warning(f"Error parsing fight row: {e}")
                continue
        
        return fights
    
    # Helper methods for parsing
    def _parse_height(self, height_str: str) -> Optional[float]:
        """Convert height string like '5\' 11\"' to cm"""
        if not height_str or height_str == '--':
            return None
        match = re.search(r"(\d+)'\s*(\d+)", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return round((feet * 12 + inches) * 2.54, 1)
        return None
    
    def _parse_weight(self, weight_str: str) -> Optional[int]:
        """Extract weight in lbs"""
        if not weight_str or weight_str == '--':
            return None
        match = re.search(r'(\d+)', weight_str)
        return int(match.group(1)) if match else None
    
    def _parse_reach(self, reach_str: str) -> Optional[float]:
        """Extract reach in inches"""
        if not reach_str or reach_str == '--':
            return None
        match = re.search(r'(\d+)', reach_str)
        return float(match.group(1)) if match else None
    
    def _parse_stat_value(self, value_str: str) -> Optional[float]:
        """Parse stat value, handling percentages and numbers"""
        if not value_str or value_str == '--':
            return None
        
        # Handle percentages
        if '%' in value_str:
            return float(value_str.strip('%')) / 100.0
        
        # Handle regular numbers
        try:
            return float(value_str)
        except ValueError:
            return None
    
    def _parse_int(self, value_str: str) -> Optional[int]:
        """Parse integer value"""
        if not value_str or value_str == '--':
            return None
        try:
            return int(value_str)
        except ValueError:
            return None
    
    def _calculate_age(self, dob_str: str) -> Optional[int]:
        """Calculate age from date of birth string"""
        if not dob_str or dob_str == '--':
            return None
        try:
            dob = datetime.strptime(dob_str.strip(), '%b %d, %Y')
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except ValueError:
            return None


def main():
    """Main function for running the scraper from command line"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Scrape UFC fighter data')
    parser.add_argument('--mode', choices=['all', 'single', 'test'], default='all',
                       help='Scrape all fighters, single fighter, or test with letter A')
    parser.add_argument('--fighter-id', type=str,
                       help='Fighter ID for single fighter mode')
    parser.add_argument('--output', type=str, default='data/processed/fighters.json',
                       help='Output file for scraped data')
    
    args = parser.parse_args()
    
    scraper = FighterScraper()
    
    if args.mode == 'all' or args.mode == 'test':
        # Get all fighter links
        test_mode = (args.mode == 'test')
        fighter_links = scraper.get_all_fighter_links(test_mode=test_mode)
        
        # Scrape each fighter
        all_fighters = []
        for i, fighter_info in enumerate(fighter_links, 1):
            logger.info(f"Scraping fighter {i}/{len(fighter_links)}: {fighter_info['name']}")
            fighter_data = scraper.scrape_fighter(fighter_info['url'], fighter_info['fighter_id'])
            if fighter_data:
                all_fighters.append(fighter_data)
            
            # Save progress every 50 fighters
            if i % 50 == 0:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(all_fighters, f, indent=2)
                logger.info(f"Progress saved: {len(all_fighters)} fighters")
        
        # Final save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_fighters, f, indent=2)
        
        logger.success(f"Scraped {len(all_fighters)} fighters. Saved to {args.output}")
    
    elif args.mode == 'single':
        if not args.fighter_id:
            logger.error("--fighter-id required for single fighter mode")
            return
        
        fighter_url = f"{scraper.base_url}/fighter-details/{args.fighter_id}"
        fighter_data = scraper.scrape_fighter(fighter_url, args.fighter_id)
        
        if fighter_data:
            print(json.dumps(fighter_data, indent=2))


if __name__ == '__main__':
    main()

