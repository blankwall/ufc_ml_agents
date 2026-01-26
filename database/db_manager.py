"""
Database Manager - Handles all database operations
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Optional
from datetime import datetime
import yaml
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Try relative import first, fall back to absolute
try:
    from .schema import (
        Base, Fighter, Event, Fight, FightStats, 
        Prediction, BettingOdds, BettingRecommendation, ModelPerformance,
        create_all_tables, drop_all_tables
    )
except ImportError:
    from database.schema import (
        Base, Fighter, Event, Fight, FightStats, 
        Prediction, BettingOdds, BettingRecommendation, ModelPerformance,
        create_all_tables, drop_all_tables
    )


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize database connection"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config['database']
        
        if db_config['type'] == 'sqlite':
            db_path = Path(db_config['sqlite_path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
        elif db_config['type'] == 'postgresql':
            pg_config = db_config['postgresql']
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}@"
                f"{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            )
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")
        
        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        create_all_tables(self.engine)
        
        logger.info(f"Database initialized: {connection_string}")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def _parse_height(self, height_str: str) -> Optional[float]:
        """Parse height string like '5\' 8"' to cm"""
        if not height_str or height_str == '--':
            return None
        try:
            # Format: 5' 8"
            parts = height_str.replace('"', '').replace("'", "").split()
            if len(parts) == 2:
                feet = int(parts[0])
                inches = int(parts[1])
                total_inches = (feet * 12) + inches
                return round(total_inches * 2.54, 1)  # Convert to cm
        except:
            pass
        return None
    
    def _parse_weight(self, weight_str: str) -> Optional[float]:
        """Parse weight string like '155 lbs.' to pounds"""
        if not weight_str or weight_str == '--':
            return None
        try:
            # Extract numeric part
            weight = weight_str.replace('lbs.', '').replace('lbs', '').strip()
            return float(weight)
        except:
            pass
        return None
    
    def _parse_reach(self, reach_str: str) -> Optional[float]:
        """Parse reach string like '68"' to inches"""
        if not reach_str or reach_str == '--':
            return None
        try:
            reach = reach_str.replace('"', '').replace("'", "").strip()
            return float(reach)
        except:
            pass
        return None
    
    def _map_fighter_data(self, fighter_data: Dict) -> Dict:
        """
        Map scraped fighter data to database schema fields
        
        Args:
            fighter_data: Raw scraped fighter data
            
        Returns:
            Mapped data matching Fighter schema
        """
        mapped = {}
        
        # Direct mappings
        direct_fields = ['fighter_id', 'name', 'nickname', 'stance', 'date_of_birth', 
                        'age', 'wins', 'losses', 'draws', 'no_contests', 'url']
        for field in direct_fields:
            if field in fighter_data:
                mapped[field] = fighter_data[field]
        
        # Physical attributes - parse and convert
        if 'height' in fighter_data:
            height_cm = self._parse_height(fighter_data['height'])
            if height_cm is not None:
                mapped['height_cm'] = height_cm
        
        if 'weight' in fighter_data:
            weight_lbs = self._parse_weight(fighter_data['weight'])
            if weight_lbs is not None:
                mapped['weight_lbs'] = weight_lbs
        
        if 'reach' in fighter_data:
            reach_inches = self._parse_reach(fighter_data['reach'])
            if reach_inches is not None:
                mapped['reach_inches'] = reach_inches
        
        # Career statistics
        stats_mapping = {
            'sig_strikes_landed_per_min': 'sig_strikes_landed_per_min',
            'striking_accuracy': 'striking_accuracy',
            'sig_strikes_absorbed_per_min': 'sig_strikes_absorbed_per_min',
            'striking_defense': 'striking_defense',
            'takedown_avg_per_15min': 'takedown_avg_per_15min',
            'takedown_accuracy': 'takedown_accuracy',
            'takedown_defense': 'takedown_defense',
            'submission_avg_per_15min': 'submission_avg_per_15min'
        }
        for json_field, db_field in stats_mapping.items():
            if json_field in fighter_data:
                mapped[db_field] = fighter_data[json_field]
        
        # Parse scraped_at timestamp
        if 'scraped_at' in fighter_data and fighter_data['scraped_at']:
            try:
                if isinstance(fighter_data['scraped_at'], str):
                    mapped['scraped_at'] = datetime.fromisoformat(fighter_data['scraped_at'].replace('Z', '+00:00'))
                else:
                    mapped['scraped_at'] = fighter_data['scraped_at']
            except Exception:
                mapped['scraped_at'] = datetime.now()
        
        return mapped
    
    def add_fighter(self, session: Session, fighter_data: Dict) -> Fighter:
        """
        Add or update a fighter in the database
        
        Args:
            session: Database session
            fighter_data: Dictionary with fighter information
            
        Returns:
            Fighter object
        """
        fighter_id = fighter_data.get('fighter_id')
        
        # Check if fighter already exists
        fighter = session.query(Fighter).filter_by(fighter_id=fighter_id).first()
        
        # Map data to schema
        mapped_data = self._map_fighter_data(fighter_data)
        
        if fighter:
            # Update existing fighter
            for key, value in mapped_data.items():
                if hasattr(fighter, key):
                    setattr(fighter, key, value)
            fighter.updated_at = datetime.now()
            logger.debug(f"Updated fighter: {fighter.name}")
        else:
            # Create new fighter
            fighter = Fighter(**mapped_data)
            session.add(fighter)
            logger.debug(f"Added new fighter: {fighter.name}")
        
        return fighter
    
    def _map_event_data(self, event_data: Dict) -> Dict:
        """
        Map scraped event data to database schema fields
        
        Args:
            event_data: Raw scraped event data
            
        Returns:
            Mapped data matching Event schema
        """
        mapped = {}
        
        # Direct mappings (excluding 'fights')
        direct_fields = ['event_id', 'name', 'date', 'location', 'venue', 'url']
        for field in direct_fields:
            if field in event_data:
                mapped[field] = event_data[field]
        
        # Parse scraped_at timestamp
        if 'scraped_at' in event_data and event_data['scraped_at']:
            try:
                if isinstance(event_data['scraped_at'], str):
                    mapped['scraped_at'] = datetime.fromisoformat(event_data['scraped_at'].replace('Z', '+00:00'))
                else:
                    mapped['scraped_at'] = event_data['scraped_at']
            except Exception:
                mapped['scraped_at'] = datetime.now()
        
        return mapped
    
    def add_event(self, session: Session, event_data: Dict) -> Event:
        """
        Add or update an event in the database
        
        Args:
            session: Database session
            event_data: Dictionary with event information
            
        Returns:
            Event object
        """
        event_id = event_data.get('event_id')
        
        # Check if event already exists
        event = session.query(Event).filter_by(event_id=event_id).first()
        
        # Map data to schema
        mapped_data = self._map_event_data(event_data)
        
        if event:
            # Update existing event
            for key, value in mapped_data.items():
                if hasattr(event, key):
                    setattr(event, key, value)
            logger.debug(f"Updated event: {event.name}")
        else:
            # Create new event
            event = Event(**mapped_data)
            session.add(event)
            logger.debug(f"Added new event: {event.name}")
        
        return event
    
    def add_fight(self, session: Session, fight_data: Dict, event: Event, 
                  fighter_1: Fighter, fighter_2: Fighter) -> Fight:
        """
        Add or update a fight in the database
        
        Args:
            session: Database session
            fight_data: Dictionary with fight information
            event: Event object
            fighter_1: First fighter object
            fighter_2: Second fighter object
            
        Returns:
            Fight object
        """
        fight_id = fight_data.get('fight_detail_id')
        
        if fight_id:
            fight = session.query(Fight).filter_by(fight_id=fight_id).first()
        else:
            # Try to find by event and fighters
            fight = session.query(Fight).filter_by(
                event_id=event.id,
                fighter_1_id=fighter_1.id,
                fighter_2_id=fighter_2.id
            ).first()
        
        # from pprint import pprint
        # pprint(fight_data)
        # pprint(fight)
        # exit()
        if fight:
            # Update existing fight
            fight.weight_class = fight_data.get('weight_class')
            fight.is_title_fight = fight_data.get('is_title_fight', False)
            fight.result = fight_data.get('result')
            fight.method = fight_data.get('method')
            fight.method_detail = fight_data.get('method_detail')
            fight.round_finished = fight_data.get('round')
            fight.time = fight_data.get('time')
            logger.debug(f"Updated fight: {fighter_1.name} vs {fighter_2.name}")
        else:
            # Create new fight
            fight = Fight(
                fight_id=fight_id,
                event_id=event.id,
                fighter_1_id=fighter_1.id,
                fighter_2_id=fighter_2.id,
                fight_number=fight_data.get('fight_number'),
                weight_class=fight_data.get('weight_class'),
                is_title_fight=fight_data.get('is_title_fight', False),
                result=fight_data.get('result'),
                method=fight_data.get('method'),
                method_detail=fight_data.get('method_detail'),
                round_finished=fight_data.get('round'),
                time=fight_data.get('time'),
                fight_detail_url=fight_data.get('fight_detail_url')
            )
            session.add(fight)
            logger.debug(f"Added new fight: {fighter_1.name} vs {fighter_2.name}")
        
        # Set winner
        if fight.result == 'fighter_1':
            fight.winner_id = fighter_1.id
        elif fight.result == 'fighter_2':
            fight.winner_id = fighter_2.id
        
        return fight
    
    def populate_from_scraped_data(self, fighters_file: str = None, events_file: str = None, fight_details_file: str = None):
        """
        Populate database from scraped JSON files
        
        Args:
            fighters_file: Path to fighters JSON file
            events_file: Path to events JSON file
            fight_details_file: Path to fight details JSON file
        """
        session = self.get_session()
        
        # Load winner lookup file (pre-processed from fight_details.json)
        winner_lookup = {}
        winner_lookup_file = 'data/processed/winner_lookup.json'
        if Path(winner_lookup_file).exists():
            logger.info(f"Loading winner lookup from {winner_lookup_file}")
            with open(winner_lookup_file, 'r') as f:
                winner_lookup = json.load(f)
            logger.success(f"Loaded {len(winner_lookup)} winner mappings")
        else:
            logger.warning(f"Winner lookup file not found: {winner_lookup_file}")
            logger.warning("Run: python create_winner_lookup.py")
            logger.warning("Proceeding without winner corrections...")
        
        try:
            # Load fighters
            if fighters_file and Path(fighters_file).exists():
                logger.info(f"Loading fighters from {fighters_file}")
                with open(fighters_file, 'r') as f:
                    fighters_data = json.load(f)
                
                for i, fighter_data in enumerate(fighters_data, 1):
                    try:
                        self.add_fighter(session, fighter_data)
                        if i % 100 == 0:
                            session.commit()
                            logger.info(f"Processed {i}/{len(fighters_data)} fighters")
                    except Exception as e:
                        logger.error(f"Error adding fighter: {e}")
                        session.rollback()
                
                session.commit()
                logger.success(f"Loaded {len(fighters_data)} fighters")
            
            # Load events and fights
            if events_file and Path(events_file).exists():
                logger.info(f"Loading events from {events_file}")
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                
                # Track winner corrections
                winner_correction_stats = {
                    'total_fights': 0,
                    'with_detail_id': 0,
                    'corrected_to_fighter_1': 0,
                    'corrected_to_fighter_2': 0,
                    'draw_or_nc': 0,
                    'name_mismatch': 0,
                    'no_detail_data': 0
                }
                
                for i, event_data in enumerate(events_data, 1):
                    try:
                        # Add event
                        event = self.add_event(session, event_data)
                        session.flush()
                        
                        # Add fights
                        for fight_data in event_data.get('fights', []):
                            winner_correction_stats['total_fights'] += 1
                            
                            # Get fight_detail_id to look up correct winner
                            fight_detail_id = fight_data.get('fight_detail_id')
                            
                            # Get or create fighters
                            fighter_1_id = fight_data.get('fighter_1_id')
                            fighter_2_id = fight_data.get('fighter_2_id')
                            
                            if not fighter_1_id or not fighter_2_id:
                                continue
                            
                            # Use winner lookup to correct the result
                            if fight_detail_id:
                                winner_correction_stats['with_detail_id'] += 1
                            
                            if fight_detail_id and fight_detail_id in winner_lookup:
                                winner_data = winner_lookup[fight_detail_id]
                                
                                # Get the winner ID from pre-processed lookup
                                winner_id = winner_data.get('winner_id')
                                result_type = winner_data.get('result_type')
                                
                                if result_type in ['draw', 'no_contest']:
                                    fight_data['result'] = result_type
                                    winner_correction_stats['draw_or_nc'] += 1
                                elif winner_id:
                                    # Map winner ID to position in events.json
                                    if winner_id == fighter_1_id:
                                        fight_data['result'] = 'fighter_1'
                                        winner_correction_stats['corrected_to_fighter_1'] += 1
                                    elif winner_id == fighter_2_id:
                                        fight_data['result'] = 'fighter_2'
                                        winner_correction_stats['corrected_to_fighter_2'] += 1
                                    else:
                                        winner_correction_stats['name_mismatch'] += 1
                                        logger.warning(f"Winner ID {winner_id} for fight {fight_detail_id} "
                                                     f"doesn't match either fighter in events: "
                                                     f"{fighter_1_id} or {fighter_2_id}")
                                
                                # Update method details if available
                                if winner_data.get('method'):
                                    fight_data['method'] = winner_data['method']
                                if winner_data.get('round'):
                                    fight_data['round'] = winner_data['round']
                                if winner_data.get('time'):
                                    fight_data['time'] = winner_data['time']
                            else:
                                # No winner data available for this fight
                                if fight_detail_id and winner_lookup:
                                    winner_correction_stats['no_detail_data'] += 1
                            
                            fighter_1 = session.query(Fighter).filter_by(fighter_id=fighter_1_id).first()
                            fighter_2 = session.query(Fighter).filter_by(fighter_id=fighter_2_id).first()
                            
                            # Create minimal fighter records if they don't exist
                            if not fighter_1:
                                fighter_1 = Fighter(
                                    fighter_id=fighter_1_id,
                                    name=fight_data.get('fighter_1_name', 'Unknown'),
                                    url=fight_data.get('fighter_1_url')
                                )
                                session.add(fighter_1)
                                session.flush()
                            
                            if not fighter_2:
                                fighter_2 = Fighter(
                                    fighter_id=fighter_2_id,
                                    name=fight_data.get('fighter_2_name', 'Unknown'),
                                    url=fight_data.get('fighter_2_url')
                                )
                                session.add(fighter_2)
                                session.flush()
                            
                            # Add fight with corrected data
                            self.add_fight(session, fight_data, event, fighter_1, fighter_2)
                        
                        if i % 10 == 0:
                            session.commit()
                            logger.info(f"Processed {i}/{len(events_data)} events")
                    
                    except Exception as e:
                        logger.error(f"Error adding event: {e}")
                        session.rollback()
                
                session.commit()
                logger.success(f"Loaded {len(events_data)} events")
                
                # Log winner correction statistics
                logger.info("\n" + "=" * 60)
                logger.info("WINNER CORRECTION STATISTICS")
                logger.info("=" * 60)
                logger.info(f"Total fights processed: {winner_correction_stats['total_fights']}")
                logger.info(f"Fights with detail_id: {winner_correction_stats['with_detail_id']}")
                logger.info(f"Corrected to fighter_1: {winner_correction_stats['corrected_to_fighter_1']}")
                logger.info(f"Corrected to fighter_2: {winner_correction_stats['corrected_to_fighter_2']}")
                logger.info(f"Draw/No Contest: {winner_correction_stats['draw_or_nc']}")
                logger.info(f"Name mismatches: {winner_correction_stats['name_mismatch']}")
                logger.info(f"No detail data: {winner_correction_stats['no_detail_data']}")
                logger.info("=" * 60 + "\n")
                
                # Verify we have both fighter_1 and fighter_2 wins
                f1_wins = winner_correction_stats['corrected_to_fighter_1']
                f2_wins = winner_correction_stats['corrected_to_fighter_2']
                if f2_wins > 0:
                    logger.success(f"✅ Winner correction working! {f2_wins} fighter_2 wins found ({f2_wins/(f1_wins+f2_wins)*100:.1f}% of total wins)")
                else:
                    logger.error("❌ WARNING: No fighter_2 wins found! All fights marked as fighter_1 wins!")
            
            # Load fight details and stats
            if fight_details_file and Path(fight_details_file).exists():
                logger.info(f"Loading fight stats from {fight_details_file}")
                with open(fight_details_file, 'r') as f:
                    fight_details_data = json.load(f)
                
                for i, details in enumerate(fight_details_data, 1):
                    try:
                        # Find the corresponding fight in database
                        fight_id = details.get('fight_id')
                        if not fight_id:
                            continue
                        
                        fight = session.query(Fight).filter_by(fight_id=fight_id).first()
                        
                        if not fight:
                            logger.debug(f"Fight {fight_id} not found in database")
                            continue
                        
                        # Check if stats already exist
                        existing_stats = session.query(FightStats).filter_by(fight_id=fight.id).first()
                        
                        if existing_stats:
                            # Update existing stats
                            existing_stats.fighter_1_totals = details.get('totals', {}).get('fighter_1')
                            existing_stats.fighter_2_totals = details.get('totals', {}).get('fighter_2')
                            existing_stats.significant_strikes = details.get('significant_strikes')
                        else:
                            # Create new stats
                            fight_stats = FightStats(
                                fight_id=fight.id,
                                fighter_1_totals=details.get('totals', {}).get('fighter_1'),
                                fighter_2_totals=details.get('totals', {}).get('fighter_2'),
                                significant_strikes=details.get('significant_strikes')
                            )
                            session.add(fight_stats)
                        
                        if i % 100 == 0:
                            session.commit()
                            logger.info(f"Processed {i}/{len(fight_details_data)} fight details")
                    
                    except Exception as e:
                        logger.error(f"Error adding fight details: {e}")
                        session.rollback()
                
                session.commit()
                logger.success(f"Loaded {len(fight_details_data)} fight details with correct winners")
        
        finally:
            session.close()
    
    def get_fighter_by_id(self, fighter_id: str) -> Optional[Fighter]:
        """Get fighter by fighter_id"""
        session = self.get_session()
        try:
            return session.query(Fighter).filter_by(fighter_id=fighter_id).first()
        finally:
            session.close()
    
    def get_fighter_stats(self, fighter_id: int) -> Dict:
        """Get comprehensive stats for a fighter"""
        session = self.get_session()
        try:
            fighter = session.query(Fighter).filter_by(id=fighter_id).first()
            if not fighter:
                return None
            
            # Get all fights
            fights = session.query(Fight).filter(
                (Fight.fighter_1_id == fighter_id) | (Fight.fighter_2_id == fighter_id)
            ).all()
            
            return {
                'fighter': fighter,
                'total_fights': len(fights),
                'fights': fights
            }
        finally:
            session.close()
    
    def get_upcoming_fights(self) -> List[Fight]:
        """Get fights that haven't happened yet (no result)"""
        session = self.get_session()
        try:
            return session.query(Fight).filter(Fight.result == None).all()
        finally:
            session.close()
    
    def add_prediction(self, session: Session, prediction_data: Dict) -> Prediction:
        """Add a prediction to the database"""
        prediction = Prediction(**prediction_data)
        session.add(prediction)
        return prediction
    
    def add_betting_odds(self, session: Session, odds_data: Dict) -> BettingOdds:
        """Add betting odds to the database"""
        odds = BettingOdds(**odds_data)
        session.add(odds)
        return odds
    
    def get_stats_summary(self) -> Dict:
        """Get database statistics summary"""
        session = self.get_session()
        try:
            num_fighters = session.query(Fighter).count()
            num_events = session.query(Event).count()
            num_fights = session.query(Fight).count()
            num_predictions = session.query(Prediction).count()
            
            return {
                'fighters': num_fighters,
                'events': num_events,
                'fights': num_fights,
                'predictions': num_predictions
            }
        finally:
            session.close()
    
    def reset_database(self):
        """Drop all tables and recreate them (WARNING: Deletes all data!)"""
        logger.warning("Resetting database - ALL DATA WILL BE LOST!")
        drop_all_tables(self.engine)
        create_all_tables(self.engine)
        logger.info("Database reset complete")


def main():
    """Main function for command-line database operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='UFC Database Manager')
    parser.add_argument('--populate', action='store_true',
                       help='Populate database from scraped data')
    parser.add_argument('--fighters-file', type=str,
                       default='data/processed/fighters.json',
                       help='Path to fighters JSON file')
    parser.add_argument('--events-file', type=str,
                       default='data/processed/events.json',
                       help='Path to events JSON file')
    parser.add_argument('--fight-details-file', type=str,
                       default='data/processed/fight_details.json',
                       help='Path to fight details JSON file')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--reset', action='store_true',
                       help='Reset database (WARNING: deletes all data)')
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    
    if args.reset:
        confirm = input("Are you sure you want to reset the database? (yes/no): ")
        if confirm.lower() == 'yes':
            db_manager.reset_database()
        else:
            logger.info("Reset cancelled")
    
    if args.populate:
        db_manager.populate_from_scraped_data(
            fighters_file=args.fighters_file,
            events_file=args.events_file,
            fight_details_file=args.fight_details_file
        )
    
    if args.stats:
        stats = db_manager.get_stats_summary()
        logger.info("Database Statistics:")
        logger.info(f"  Fighters: {stats['fighters']}")
        logger.info(f"  Events: {stats['events']}")
        logger.info(f"  Fights: {stats['fights']}")
        logger.info(f"  Predictions: {stats['predictions']}")


if __name__ == '__main__':
    main()

