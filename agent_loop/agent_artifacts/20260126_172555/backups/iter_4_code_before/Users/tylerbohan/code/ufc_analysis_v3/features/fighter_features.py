"""
Fighter Feature Extraction - Creates predictive features from fighter statistics

REFACTORED: Now uses modular feature system with FeatureBuilder and FeatureRegistry.
This provides a clean, maintainable structure where features can be easily toggled
on/off and new features can be added without modifying core extraction logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from database.schema import Fighter, Fight, FightStats
from .registry import FeatureBuilder, FeatureRegistry
from .feature_toggles import DEFAULT_FEATURE_SET


class FighterFeatureExtractor:
    """
    Extract features for individual fighters.
    
    This class now acts as a thin wrapper around the modular FeatureBuilder system,
    maintaining backward compatibility while using the new architecture.
    """
    
    def __init__(self, session: Session, rolling_windows: List[int] = [3, 5]):
        """
        Initialize feature extractor.
        
        Args:
            session: Database session
            rolling_windows: Windows for calculating rolling statistics
        """
        self.session = session
        self.rolling_windows = rolling_windows
        self.feature_builder = FeatureBuilder(session, rolling_windows)
    
    def extract_features(
        self,
        fighter_id: int,
        as_of_date: Optional[Union[datetime, str]] = None,
        feature_set: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract all features for a fighter.
        
        Args:
            fighter_id: Fighter database ID
            as_of_date: Calculate features as of this date (for historical analysis)
            feature_set: Optional list of feature group names to extract.
                        If None, uses the toggled DEFAULT_FEATURE_SET from
                        `features/feature_toggles.py`.
            
        Returns:
            Dictionary of features
        """
        # Use toggle-based feature set by default
        if feature_set is None:
            feature_set = list(DEFAULT_FEATURE_SET)
        
        return self.feature_builder.build_features(
            fighter_id=fighter_id,
            feature_set=feature_set,
            as_of_date=as_of_date
        )
    
    # Backward compatibility: expose old method names as properties/methods
    def _get_fight_history(
        self,
        fighter_id: int,
        as_of_date: Optional[Union[datetime, str]] = None
    ) -> pd.DataFrame:
        """
        Get fight history for a fighter.
        
        This method is kept for backward compatibility with code that might
        call it directly. New code should use FeatureBuilder.get_fight_history.
        """
        return self.feature_builder.get_fight_history(fighter_id, as_of_date)
    
    def _get_fighter_record_from_cache(self, fighter_id: int) -> Optional[Dict]:
        """
        Get a simple record summary for a fighter, with basic caching.
        
        This method is kept for backward compatibility. New code should use
        FeatureBuilder.get_fighter_record.
        """
        return self.feature_builder.get_fighter_record(fighter_id)


def extract_features_for_all_fighters(
    session: Session,
    feature_set: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract features for all fighters in the database.
    
    Args:
        session: Database session
        feature_set: Optional list of feature names to extract.
                    If None, uses FEATURE_SET_FULL (all features)
    
    Returns:
        DataFrame with features for all fighters
    """
    extractor = FighterFeatureExtractor(session)
    
    if feature_set is None:
        feature_set = FeatureRegistry.FEATURE_SET_FULL
    
    fighters = session.query(Fighter).all()
    all_features = []
    
    for i, fighter in enumerate(fighters, 1):
        if i % 100 == 0:
            logger.info(f"Extracting features for fighter {i}/{len(fighters)}")
        
        features = extractor.extract_features(fighter.id, feature_set=feature_set)
        features['fighter_id'] = fighter.id
        features['fighter_name'] = fighter.name
        all_features.append(features)
    
    df = pd.DataFrame(all_features)
    logger.success(f"Extracted features for {len(df)} fighters")
    
    return df


if __name__ == '__main__':
    import argparse
    from database.db_manager import DatabaseManager
    
    parser = argparse.ArgumentParser(description='Extract features for all fighters')
    parser.add_argument('--feature-set', type=str,
                       choices=['base', 'advanced', 'full'],
                       default='full',
                       help='Feature set to use: base, advanced, or full (default: full)')
    parser.add_argument('--output', type=str,
                       default='data/processed/fighter_features.csv',
                       help='Output path for fighter features CSV')
    
    args = parser.parse_args()
    
    # Map feature set string to actual feature set
    feature_set_map = {
        'base': FeatureRegistry.FEATURE_SET_BASE,
        'advanced': FeatureRegistry.FEATURE_SET_ADVANCED,
        'full': FeatureRegistry.FEATURE_SET_FULL,
    }
    feature_set = feature_set_map[args.feature_set]
    
    db = DatabaseManager()
    session = db.get_session()
    
    logger.info(f"Extracting features with '{args.feature_set}' feature set...")
    df = extract_features_for_all_fighters(session, feature_set=feature_set)
    
    # Save to file
    df.to_csv(args.output, index=False)
    logger.info(f"Saved fighter features to {args.output}")
    logger.info(f"Extracted {len(df)} fighters with {len(df.columns)} features")
    
    session.close()
