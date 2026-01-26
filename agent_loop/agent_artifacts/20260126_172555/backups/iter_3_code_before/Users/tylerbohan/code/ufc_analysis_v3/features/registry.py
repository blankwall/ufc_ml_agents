"""
Feature Registry and Orchestrator
Central system for assembling and managing feature sets
"""

import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
from datetime import datetime
from loguru import logger

from database.schema import Fighter, FightStats
from sqlalchemy.orm import Session

# Import all feature modules
from .physical import extract_physical_features
from .striking import extract_striking_features, extract_recent_striking_features
from .grappling import extract_grappling_features, extract_recent_grappling_features
from .experiential import (
    extract_career_stats,
    extract_fight_history_features,
    extract_early_finishing_features,
    extract_round_3_features,
)
from .time_based import (
    extract_rolling_stats,
    extract_momentum_features,
    extract_decline_features,
    extract_recent_damage_features,
    extract_time_decayed_features,
    extract_opponent_quality_adjusted_time_decayed_features,
    extract_recent_form_quality_adjusted,
    extract_age_interactions,
    extract_youth_form_score,
    extract_prospect_momentum_score,
    extract_early_finish_advantage,
    extract_power_striker_score,
    extract_age_weighted_recent_damage,
    extract_durability_collapse_score,
)
from .opponent_quality import extract_opponent_quality_features


# Feature function type hint
FeatureFunction = Callable[[Any], Dict[str, float]]


class FeatureRegistry:
    """
    Registry of all available feature extraction functions.
    
    This allows easy composition of feature sets and toggling features on/off.
    """
    
    # Base feature sets
    FEATURE_SET_PHYSICAL = [
        "physical",
    ]
    
    FEATURE_SET_STRIKING = [
        "striking",
    ]
    
    FEATURE_SET_GRAPPLING = [
        "grappling",
    ]
    
    FEATURE_SET_EXPERIENTIAL = [
        "career_stats",
        "fight_history",
        "early_finishing",
        "round_3",
    ]
    
    FEATURE_SET_TIME_BASED = [
        "rolling_stats",
        "momentum",
        "decline",
        "recent_damage",
        "time_decayed",
        "time_decayed_adj_opp_quality",
        "recent_form_quality_adjusted",
    ]
    
    FEATURE_SET_OPPONENT_QUALITY = [
        "opponent_quality",
    ]
    
    FEATURE_SET_INTERACTIONS = [
        "age_interactions",
        "youth_form",
        "prospect_momentum",
        "early_finish_advantage",
        "power_striker",
    ]
    
    FEATURE_SET_RECENT_STATS = [
        "recent_striking",
        "recent_grappling",
    ]
    
    # Predefined feature set combinations
    FEATURE_SET_BASE = (
        FEATURE_SET_PHYSICAL +
        FEATURE_SET_STRIKING +
        FEATURE_SET_GRAPPLING +
        FEATURE_SET_EXPERIENTIAL +
        FEATURE_SET_TIME_BASED
    )
    
    FEATURE_SET_ADVANCED = (
        FEATURE_SET_BASE +
        FEATURE_SET_OPPONENT_QUALITY +
        FEATURE_SET_INTERACTIONS
    )
    
    FEATURE_SET_FULL = (
        FEATURE_SET_ADVANCED +
        FEATURE_SET_RECENT_STATS
    )
    
    @classmethod
    def get_feature_function(cls, feature_name: str) -> Optional[FeatureFunction]:
        """
        Get a feature extraction function by name.
        
        Args:
            feature_name: Name of the feature group
            
        Returns:
            Feature extraction function or None if not found
        """
        feature_map = {
            "physical": cls._extract_physical,
            "striking": cls._extract_striking,
            "grappling": cls._extract_grappling,
            "career_stats": cls._extract_career_stats,
            "fight_history": cls._extract_fight_history,
            "early_finishing": cls._extract_early_finishing,
            "round_3": cls._extract_round_3,
            "rolling_stats": cls._extract_rolling_stats,
            "momentum": cls._extract_momentum,
            "decline": cls._extract_decline,
            "recent_damage": cls._extract_recent_damage,
            "time_decayed": cls._extract_time_decayed,
            "time_decayed_adj_opp_quality": cls._extract_time_decayed_adj_opp_quality,
            "recent_form_quality_adjusted": cls._extract_recent_form_quality_adjusted,
            "opponent_quality": cls._extract_opponent_quality,
            "age_interactions": cls._extract_age_interactions,
            "youth_form": cls._extract_youth_form,
            "prospect_momentum": cls._extract_prospect_momentum,
            "early_finish_advantage": cls._extract_early_finish_advantage,
            "power_striker": cls._extract_power_striker,
            "recent_striking": cls._extract_recent_striking,
            "recent_grappling": cls._extract_recent_grappling,
            "age_weighted_recent_damage": cls._extract_age_weighted_recent_damage,
            "durability_collapse": cls._extract_durability_collapse,
        }
        return feature_map.get(feature_name)
    
    # Wrapper methods that adapt pure functions to the registry interface
    @staticmethod
    def _extract_physical(context: Dict) -> Dict[str, float]:
        """Extract physical features"""
        fighter = context["fighter"]
        return extract_physical_features(fighter)
    
    @staticmethod
    def _extract_striking(context: Dict) -> Dict[str, float]:
        """Extract striking features"""
        # Pass full context to ensure point-in-time safety
        return extract_striking_features(context)
    
    @staticmethod
    def _extract_grappling(context: Dict) -> Dict[str, float]:
        """Extract grappling features"""
        # Pass full context to ensure point-in-time safety
        return extract_grappling_features(context)
    
    @staticmethod
    def _extract_career_stats(context: Dict) -> Dict[str, float]:
        """Extract career statistics"""
        fighter = context["fighter"]
        fight_history = context["fight_history"]
        return extract_career_stats(fighter, fight_history)
    
    @staticmethod
    def _extract_fight_history(context: Dict) -> Dict[str, float]:
        """Extract fight history features"""
        fight_history = context["fight_history"]
        return extract_fight_history_features(fight_history)
    
    @staticmethod
    def _extract_early_finishing(context: Dict) -> Dict[str, float]:
        """Extract early finishing features"""
        fight_history = context["fight_history"]
        return extract_early_finishing_features(fight_history)
    
    @staticmethod
    def _extract_round_3(context: Dict) -> Dict[str, float]:
        """Extract round 3 features"""
        fight_history = context["fight_history"]
        return extract_round_3_features(fight_history)
    
    @staticmethod
    def _extract_rolling_stats(context: Dict) -> Dict[str, float]:
        """Extract rolling statistics"""
        fight_history = context["fight_history"]
        rolling_windows = context.get("rolling_windows", [3, 5])
        return extract_rolling_stats(fight_history, rolling_windows)
    
    @staticmethod
    def _extract_momentum(context: Dict) -> Dict[str, float]:
        """Extract momentum features"""
        fight_history = context["fight_history"]
        return extract_momentum_features(fight_history, as_of_date=context.get("as_of_date"))
    
    @staticmethod
    def _extract_decline(context: Dict) -> Dict[str, float]:
        """Extract decline features"""
        fight_history = context["fight_history"]
        return extract_decline_features(fight_history, as_of_date=context.get("as_of_date"))
    
    @staticmethod
    def _extract_recent_damage(context: Dict) -> Dict[str, float]:
        """Extract recent damage features"""
        fight_history = context["fight_history"]
        return extract_recent_damage_features(fight_history)
    
    @staticmethod
    def _extract_time_decayed(context: Dict) -> Dict[str, float]:
        """Extract time-decayed features"""
        fight_history = context["fight_history"]
        lambda_decay = context.get("lambda_decay", 0.3)
        return extract_time_decayed_features(
            fight_history,
            lambda_decay,
            as_of_date=context.get("as_of_date"),
        )
    
    @staticmethod
    def _extract_time_decayed_adj_opp_quality(context: Dict) -> Dict[str, float]:
        """Extract opponent-quality-adjusted time-decayed features"""
        fight_history = context["fight_history"]
        get_fighter_record = context["get_fighter_record"]
        lambda_decay = context.get("lambda_decay", 0.3)
        return extract_opponent_quality_adjusted_time_decayed_features(
            fight_history, get_fighter_record, lambda_decay
        )
    
    @staticmethod
    def _extract_recent_form_quality_adjusted(context: Dict) -> Dict[str, float]:
        """Extract recent form quality-adjusted features"""
        fight_history = context["fight_history"]
        get_fighter_record = context["get_fighter_record"]
        window = context.get("recent_form_window", 5)
        return extract_recent_form_quality_adjusted(
            fight_history, get_fighter_record, window
        )
    
    @staticmethod
    def _extract_opponent_quality(context: Dict) -> Dict[str, float]:
        """Extract opponent quality features"""
        fight_history = context["fight_history"]
        get_fighter_record = context["get_fighter_record"]
        return extract_opponent_quality_features(fight_history, get_fighter_record)
    
    @staticmethod
    def _extract_age_interactions(context: Dict) -> Dict[str, float]:
        """Extract age interaction features"""
        physical_features = context.get("physical_features", {})
        decline_features = context.get("decline_features", {})
        momentum_features = context.get("momentum_features", {})
        
        age = physical_features.get("age", 0.0)
        return extract_age_interactions(age, decline_features, momentum_features)
    
    @staticmethod
    def _extract_youth_form(context: Dict) -> Dict[str, float]:
        """Extract youth form score"""
        physical_features = context.get("physical_features", {})
        rolling_features = context.get("rolling_features", {})
        
        age = physical_features.get("age", 0.0)
        win_rate_last_5 = rolling_features.get("win_rate_last_5", 0.0)
        finish_rate_last_5 = rolling_features.get("finish_rate_last_5", 0.0)
        
        score = extract_youth_form_score(age, win_rate_last_5, finish_rate_last_5)
        return {"youth_form_score": score}
    
    @staticmethod
    def _extract_prospect_momentum(context: Dict) -> Dict[str, float]:
        """Extract prospect momentum score"""
        physical_features = context.get("physical_features", {})
        rolling_features = context.get("rolling_features", {})
        
        age = physical_features.get("age", 0.0)
        win_rate_last_5 = rolling_features.get("win_rate_last_5", 0.0)
        finish_rate_last_5 = rolling_features.get("finish_rate_last_5", 0.0)
        
        score = extract_prospect_momentum_score(age, win_rate_last_5, finish_rate_last_5)
        return {"prospect_momentum_score": score}
    
    @staticmethod
    def _extract_early_finish_advantage(context: Dict) -> Dict[str, float]:
        """Extract early finish advantage score"""
        early_finishing_features = context.get("early_finishing_features", {})
        time_decayed_features = context.get("time_decayed_features", {})
        
        first_round_finish_rate = early_finishing_features.get("first_round_finish_rate", 0.0)
        early_finish_rate_last_3 = early_finishing_features.get("early_finish_rate_last_3", 0.0)
        time_decayed_ko_rate = time_decayed_features.get("time_decayed_ko_rate", 0.0)
        
        score = extract_early_finish_advantage(
            first_round_finish_rate,
            early_finish_rate_last_3,
            time_decayed_ko_rate
        )
        return {"early_finish_advantage": score}
    
    @staticmethod
    def _extract_age_weighted_recent_damage(context: Dict) -> Dict[str, float]:
        """Extract age-weighted recent damage feature"""
        physical_features = context.get("physical_features", {})
        recent_striking_features = context.get("recent_striking_features", {})
        
        age = physical_features.get("age", 0.0)
        recent_sig_strike_diff_last_3 = recent_striking_features.get("recent_sig_strike_diff_last_3", 0.0)
        
        score = extract_age_weighted_recent_damage(age, recent_sig_strike_diff_last_3)
        return {"age_weighted_recent_damage": score}
    
    @staticmethod
    def _extract_durability_collapse(context: Dict) -> Dict[str, float]:
        """Extract durability collapse score"""
        physical_features = context.get("physical_features", {})
        recent_striking_features = context.get("recent_striking_features", {})
        recent_damage_features = context.get("recent_damage_features", {})
        rolling_features = context.get("rolling_features", {})
        
        age = physical_features.get("age", 0.0)
        recent_knockdown_diff_last_3 = recent_striking_features.get("recent_knockdown_diff_last_3", 0.0)
        recent_finish_losses_last_2 = recent_damage_features.get("recent_finish_losses_last_2", 0.0)
        athleticism_decline = rolling_features.get("athleticism_decline", 0.0)
        
        score = extract_durability_collapse_score(
            age,
            recent_knockdown_diff_last_3,
            recent_finish_losses_last_2,
            athleticism_decline
        )
        return {"durability_collapse_score": score}
    
    @staticmethod
    def _extract_power_striker(context: Dict) -> Dict[str, float]:
        """Extract power striker score"""
        rolling_features = context.get("rolling_features", {})
        early_finishing_features = context.get("early_finishing_features", {})
        striking_features = context.get("striking_features", {})
        career_stats = context.get("career_stats", {})
        
        ko_rate_last_5 = rolling_features.get("ko_rate_last_5", 0.0)
        first_round_ko_rate = early_finishing_features.get("first_round_ko_rate", 0.0)
        
        # Calculate knockdowns_per_fight from striking features
        knockdowns_lifetime = striking_features.get("knockdowns_lifetime", 0.0)
        total_fights = career_stats.get("total_fights", 0.0)
        knockdowns_per_fight = knockdowns_lifetime / total_fights if total_fights > 0 else 0.0
        
        # Use lifetime head strike rate (or last_3 if available)
        head_strike_rate = striking_features.get("head_strike_rate_lifetime", 0.0)
        if head_strike_rate == 0.0:
            head_strike_rate = striking_features.get("head_strike_rate_last_3", 0.0)
        
        score = extract_power_striker_score(
            ko_rate_last_5,
            first_round_ko_rate,
            knockdowns_per_fight,
            head_strike_rate
        )
        return {"power_striker_score": score}
    
    @staticmethod
    def _extract_recent_striking(context: Dict) -> Dict[str, float]:
        """Extract recent striking features from FightStats"""
        fight_history = context["fight_history"]
        fight_stats_by_fight_id = context.get("fight_stats_by_fight_id", {})
        fighter_id = context["fighter_id"]
        return extract_recent_striking_features(
            fight_history, fight_stats_by_fight_id, fighter_id
        )
    
    @staticmethod
    def _extract_recent_grappling(context: Dict) -> Dict[str, float]:
        """Extract recent grappling features from FightStats"""
        fight_history = context["fight_history"]
        fight_stats_by_fight_id = context.get("fight_stats_by_fight_id", {})
        fighter_id = context["fighter_id"]
        return extract_recent_grappling_features(
            fight_history, fight_stats_by_fight_id, fighter_id
        )


class FeatureBuilder:
    """
    Orchestrator for building features from a list of feature function names.
    
    This is the main interface for feature extraction - pass a list of feature
    names and it will assemble them into a complete feature dictionary.
    """
    
    def __init__(
        self,
        session: Session,
        rolling_windows: List[int] = [3, 5],
        lambda_decay: float = 0.3
    ):
        """
        Initialize feature builder.
        
        Args:
            session: Database session
            rolling_windows: Windows for rolling statistics
            lambda_decay: Decay rate for time-decayed features
        """
        self.session = session
        self.rolling_windows = rolling_windows
        self.lambda_decay = lambda_decay
        # Cache key is (fighter_id, as_of_date_iso_or_none)
        self._fighter_record_cache: Dict[Tuple[int, Optional[str]], Dict] = {}
    
    def get_fighter_record(
        self,
        fighter_id: int,
        as_of_date: Optional[Union[datetime, str]] = None
    ) -> Optional[Dict]:
        """
        Get fighter record with optional point-in-time calculation.
        
        Args:
            fighter_id: Fighter database ID
            as_of_date: Calculate record as of this date (None = all time)
            
        Returns:
            Dictionary with wins, losses, draws, total_fights, win_rate
        """
        # Create cache key that includes as_of_date
        # Handle both datetime objects and strings
        if as_of_date is None:
            cache_key_date = None
        elif isinstance(as_of_date, str):
            cache_key_date = as_of_date
        else:
            cache_key_date = as_of_date.isoformat()
        
        cache_key = (fighter_id, cache_key_date)
        
        if cache_key in self._fighter_record_cache:
            return self._fighter_record_cache[cache_key]
        
        # Calculate from fight history to respect as_of_date
        fight_history = self.get_fight_history(fighter_id, as_of_date)
        
        if len(fight_history) == 0:
            # No fight history available
            record = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_fights": 0,
                "win_rate": 0.0,
            }
            self._fighter_record_cache[cache_key] = record
            return record
        
        wins = (fight_history["result"] == "win").sum()
        losses = (fight_history["result"] == "loss").sum()
        draws = (fight_history["result"] == "draw").sum()
        total_fights = len(fight_history)
        win_rate = wins / total_fights if total_fights > 0 else 0.0
        
        record = {
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "total_fights": total_fights,
            "win_rate": float(win_rate),
        }
        self._fighter_record_cache[cache_key] = record
        return record
    
    def get_fight_history(
        self,
        fighter_id: int,
        as_of_date: Optional[Union[datetime, str]] = None
    ) -> pd.DataFrame:
        """
        Get fight history for a fighter.
        
        Args:
            fighter_id: Fighter database ID
            as_of_date: Calculate features as of this date
            
        Returns:
            DataFrame with fight history (sorted most recent first)
        """
        from database.schema import Fight
        
        fights = self.session.query(Fight).filter(
            (Fight.fighter_1_id == fighter_id) | (Fight.fighter_2_id == fighter_id)
        ).join(Fight.event).all()
        
        if not fights:
            return pd.DataFrame()
        
        fight_records = []
        for fight in fights:
            is_fighter_1 = fight.fighter_1_id == fighter_id
            opponent_id = fight.fighter_2_id if is_fighter_1 else fight.fighter_1_id
            
            if fight.result == 'fighter_1' and is_fighter_1:
                result = 'win'
            elif fight.result == 'fighter_2' and not is_fighter_1:
                result = 'win'
            elif fight.result == 'draw':
                result = 'draw'
            elif fight.result == 'no_contest':
                result = 'nc'
            else:
                result = 'loss'
            
            fight_records.append({
                'fight_id': fight.id,
                'is_fighter_1': is_fighter_1,
                'event_date': fight.event.date,
                'opponent_id': opponent_id,
                'result': result,
                'method': fight.method,
                'round': fight.round_finished,
                'is_title_fight': fight.is_title_fight,
                'weight_class': fight.weight_class
            })
        
        df = pd.DataFrame(fight_records)
        
        if len(df) > 0:
            df['event_date_parsed'] = pd.to_datetime(df['event_date'])
            
            if as_of_date is not None:
                # Ensure as_of_date is a datetime for comparison
                if isinstance(as_of_date, str):
                    as_of_date_dt = pd.to_datetime(as_of_date)
                else:
                    as_of_date_dt = as_of_date
                
                # CRITICAL: Use < not <= to exclude the fight we're predicting
                df = df[df['event_date_parsed'] < as_of_date_dt]
            
            df = df.sort_values('event_date_parsed', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_fight_stats(self, fight_ids: List[int]) -> Dict[int, Any]:
        """
        Get FightStats for a list of fight IDs.
        
        Args:
            fight_ids: List of fight IDs
            
        Returns:
            Dictionary mapping fight_id to FightStats object
        """
        if not fight_ids:
            return {}
        
        stats_list = (
            self.session.query(FightStats)
            .filter(FightStats.fight_id.in_(fight_ids))
            .all()
        )
        return {s.fight_id: s for s in stats_list}
    
    def build_features(
        self,
        fighter_id: int,
        feature_set: List[str],
        as_of_date: Optional[Union[datetime, str]] = None
    ) -> Dict[str, float]:
        """
        Build features for a fighter using specified feature set.
        
        Args:
            fighter_id: Fighter database ID
            feature_set: List of feature names to extract
            as_of_date: Calculate features as of this date
            
        Returns:
            Dictionary of all extracted features
        """
        from database.schema import Fighter
        
        fighter = self.session.query(Fighter).filter_by(id=fighter_id).first()
        if not fighter:
            logger.error(f"Fighter {fighter_id} not found")
            return {}
        
        # Get fight history
        fight_history = self.get_fight_history(fighter_id, as_of_date)
        
        # Get fight stats if needed (for striking, grappling, and recent features)
        fight_stats_by_fight_id = {}
        if any(f in feature_set for f in ["striking", "recent_striking", "grappling", "recent_grappling"]):
            # Check if fight_history has the fight_id column before accessing it
            if len(fight_history) > 0 and "fight_id" in fight_history.columns:
                fight_ids = [
                    int(fid) for fid in fight_history["fight_id"].tolist()
                    if pd.notna(fid)
                ]
                fight_stats_by_fight_id = self.get_fight_stats(fight_ids)
        
        # Create a wrapped get_fighter_record that includes as_of_date
        def get_fighter_record_as_of(fid: int) -> Optional[Dict]:
            return self.get_fighter_record(fid, as_of_date)
        
        # Build context for feature extraction
        context = {
            "fighter": fighter,
            "fighter_id": fighter_id,
            "fight_history": fight_history,
            "as_of_date": as_of_date,
            "rolling_windows": self.rolling_windows,
            "lambda_decay": self.lambda_decay,
            "get_fighter_record": get_fighter_record_as_of,  # Pass wrapped version
            "fight_stats_by_fight_id": fight_stats_by_fight_id,
            "session": self.session,  # Needed for querying Fight objects in striking features
        }
        
        # Extract features in order
        all_features = {}
        
        # First pass: extract base features
        for feature_name in feature_set:
            feature_func = FeatureRegistry.get_feature_function(feature_name)
            if feature_func is None:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            try:
                features = feature_func(context)
                all_features.update(features)
                
                # Store in context for dependent features
                if feature_name == "physical":
                    context["physical_features"] = features
                elif feature_name == "rolling_stats":
                    context["rolling_features"] = features
                elif feature_name == "decline":
                    context["decline_features"] = features
                elif feature_name == "momentum":
                    context["momentum_features"] = features
                elif feature_name == "striking":
                    context["striking_features"] = features
                elif feature_name == "career_stats":
                    context["career_stats"] = features
                elif feature_name == "early_finishing":
                    context["early_finishing_features"] = features
                elif feature_name == "time_decayed":
                    context["time_decayed_features"] = features
            except Exception as e:
                logger.error(f"Error extracting feature {feature_name}: {e}")
                continue
        
        return all_features

