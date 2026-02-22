"""
Physical and Demographic Features
Age, height, weight, reach, stance, and related physical attributes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from database.schema import Fighter


def extract_physical_features(fighter: Fighter) -> Dict[str, float]:
    """
    Extract physical and demographic features from a fighter.
    
    Pure function that takes a Fighter object and returns a dictionary of features.
    
    Args:
        fighter: Fighter database object
        
    Returns:
        Dictionary of physical features with snake_case names
        Missing values are set to NaN (not 0) to avoid creating false differentials
    """
    age = fighter.age if fighter.age is not None else np.nan
    height_cm = fighter.height_cm if fighter.height_cm is not None else np.nan
    weight_lbs = fighter.weight_lbs if fighter.weight_lbs is not None else np.nan
    reach_inches = fighter.reach_inches if fighter.reach_inches is not None else np.nan
    stance = fighter.stance or ""
    
    features = {
        # Raw physical attributes
        # Use NaN for missing values to avoid creating false differentials
        "age": float(age) if not np.isnan(age) else np.nan,
        "height_cm": float(height_cm) if not np.isnan(height_cm) else np.nan,
        "weight_lbs": float(weight_lbs) if not np.isnan(weight_lbs) else np.nan,
        "reach_inches": float(reach_inches) if not np.isnan(reach_inches) else np.nan,
        
        # Age categories (binary indicators)
        # Handle NaN age gracefully
        "age_in_prime": 1.0 if not np.isnan(age) and 27 <= age <= 33 else 0.0,
        "age_past_prime": 1.0 if not np.isnan(age) and age > 34 else 0.0,
        
        # Stance indicators (one-hot encoded)
        "stance_orthodox": 1.0 if stance == "Orthodox" else 0.0,
        "stance_southpaw": 1.0 if stance == "Southpaw" else 0.0,
        "stance_switch": 1.0 if stance == "Switch" else 0.0,
    }
    
    return features

