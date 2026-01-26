"""
Feature Vector Builder - Enforces Schema Alignment

This module provides a centralized way to build feature vectors that are
guaranteed to match the canonical feature schema. This ensures:

- 100% identical feature order
- 100% identical feature count
- 0% feature drift
- XGBoost monotone constraints line up cleanly
- Excel export = CLI prediction = training run

This is the master contract between feature extraction and model prediction.
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger

from schema import get_feature_list, align_feature_vector, validate_feature_vector


def build_feature_vector(
    feature_dict: Dict[str, float],
    fill_missing: float = 0.0,
    strict: bool = False,
    schema_path: str = "schema/feature_schema.json"
) -> np.ndarray:
    """
    Build a feature vector aligned to the canonical schema.
    
    This function:
    1. Takes a raw feature dictionary (from feature extraction)
    2. Aligns it to the schema order
    3. Fills missing features with a default value
    4. Returns a perfectly ordered numpy array for XGBoost
    
    Args:
        feature_dict: Dictionary of feature names to values (from feature extraction)
        fill_missing: Value to use for missing features (default: 0.0)
        strict: If True, validate that all required features are present
        schema_path: Path to feature schema file
        
    Returns:
        Numpy array of feature values in canonical schema order
        
    Example:
        >>> from features.matchup_features import MatchupFeatureExtractor
        >>> extractor = MatchupFeatureExtractor(session)
        >>> features = extractor.extract_matchup_features(f1_id, f2_id)
        >>> feature_vector = build_feature_vector(features)
        >>> model.predict(np.array([feature_vector]))  # Always works!
    """
    # Validate feature vector structure (non-strict by default to allow missing features)
    is_valid, missing = validate_feature_vector(
        feature_dict,
        schema_path=schema_path,
        strict=strict
    )
    
    if not is_valid and strict:
        raise ValueError(
            f"Feature vector validation failed. Missing {len(missing)} required features: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    
    # Align to schema order and fill missing features
    aligned = align_feature_vector(
        feature_dict,
        schema_path=schema_path,
        fill_missing=fill_missing
    )
    
    # Convert to numpy array
    feature_array = np.array(aligned, dtype=np.float32)
    
    # Log summary
    schema_features = get_feature_list(schema_path)
    provided_features = set(feature_dict.keys())
    schema_features_set = set(schema_features)
    
    missing_count = len(schema_features_set - provided_features)
    extra_count = len(provided_features - schema_features_set)
    
    if missing_count > 0:
        logger.debug(
            f"Feature vector: {len(provided_features)} provided, "
            f"{missing_count} missing (filled with {fill_missing}), "
            f"{extra_count} extra (ignored)"
        )
    
    return feature_array


def build_feature_vector_from_matchup(
    fighter_1_id: int,
    fighter_2_id: int,
    session,
    as_of_date: Optional[datetime] = None,
    feature_set: Optional[List[str]] = None,
    is_title_fight: bool = False,
    fill_missing: float = 0.0,
    schema_path: str = "schema/feature_schema.json"
) -> np.ndarray:
    """
    Build a feature vector directly from fighter IDs.
    
    This is a convenience function that combines feature extraction and
    vector building into a single call.
    
    Args:
        fighter_1_id: First fighter database ID
        fighter_2_id: Second fighter database ID
        session: Database session
        as_of_date: Calculate features as of this date
        feature_set: Optional list of feature names to extract
        is_title_fight: Whether this is a title fight
        fill_missing: Value to use for missing features
        schema_path: Path to feature schema file
        
    Returns:
        Numpy array of feature values in canonical schema order
    """
    from .matchup_features import MatchupFeatureExtractor
    
    # Extract features
    extractor = MatchupFeatureExtractor(session)
    feature_dict = extractor.extract_matchup_features(
        fighter_1_id,
        fighter_2_id,
        as_of_date=as_of_date,
        feature_set=feature_set
    )
    
    # Add fight-specific features
    feature_dict['is_title_fight'] = 1 if is_title_fight else 0
    
    # Build aligned vector
    return build_feature_vector(
        feature_dict,
        fill_missing=fill_missing,
        strict=False,
        schema_path=schema_path
    )


def build_feature_vectors_batch(
    feature_dicts: List[Dict[str, float]],
    fill_missing: float = 0.0,
    schema_path: str = "schema/feature_schema.json"
) -> np.ndarray:
    """
    Build multiple feature vectors in batch.
    
    Args:
        feature_dicts: List of feature dictionaries
        fill_missing: Value to use for missing features
        schema_path: Path to feature schema file
        
    Returns:
        2D numpy array where each row is a feature vector
    """
    vectors = []
    for feature_dict in feature_dicts:
        vector = build_feature_vector(
            feature_dict,
            fill_missing=fill_missing,
            strict=False,
            schema_path=schema_path
        )
        vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)


def get_schema_info(schema_path: str = "schema/feature_schema.json") -> Dict:
    """
    Get information about the feature schema.
    
    Args:
        schema_path: Path to feature schema file
        
    Returns:
        Dictionary with schema information
    """
    from schema import load_schema
    
    schema = load_schema(schema_path)
    return {
        'version': schema.get('version'),
        'num_features': schema.get('num_features'),
        'feature_names': schema.get('features', [])
    }


if __name__ == '__main__':
    """Test the feature vector builder"""
    import sys
    from database.db_manager import DatabaseManager
    from database.schema import Fighter
    
    if len(sys.argv) < 3:
        print("Usage: python -m features.feature_vector_builder <fighter_1_name> <fighter_2_name>")
        sys.exit(1)
    
    fighter_1_name = sys.argv[1]
    fighter_2_name = sys.argv[2]
    
    db = DatabaseManager()
    session = db.get_session()
    
    try:
        # Find fighters
        f1 = session.query(Fighter).filter(Fighter.name.ilike(f'%{fighter_1_name}%')).first()
        f2 = session.query(Fighter).filter(Fighter.name.ilike(f'%{fighter_2_name}%')).first()
        
        if not f1:
            logger.error(f"Fighter not found: {fighter_1_name}")
            sys.exit(1)
        if not f2:
            logger.error(f"Fighter not found: {fighter_2_name}")
            sys.exit(1)
        
        logger.info(f"Building feature vector for: {f1.name} vs {f2.name}")
        
        # Build feature vector
        feature_vector = build_feature_vector_from_matchup(
            f1.id,
            f2.id,
            session,
            is_title_fight=False
        )
        
        logger.success(f"Built feature vector with shape: {feature_vector.shape}")
        logger.info(f"First 10 features: {feature_vector[:10]}")
        logger.info(f"Last 10 features: {feature_vector[-10:]}")
        
        # Show schema info
        schema_info = get_schema_info()
        logger.info(f"Schema version: {schema_info['version']}")
        logger.info(f"Total features: {schema_info['num_features']}")
        
    finally:
        session.close()

