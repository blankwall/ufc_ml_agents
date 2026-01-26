"""
Feature Schema Module

This module provides utilities for loading and working with the canonical
feature schema. The schema is the master contract between training, prediction,
Excel export, and API usage.
"""

from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
from loguru import logger


def load_schema(schema_path: str = "schema/feature_schema.json") -> Dict:
    """
    Load feature schema from file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Schema dictionary with keys: version, num_features, features
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    schema_file = Path(schema_path)
    
    if not schema_file.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_file}\n"
            "Generate the schema first using:\n"
            "  python schema/generate_schema.py\n"
            "or\n"
            "  model.export_feature_schema()"
        )
    
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    
    logger.debug(f"Loaded schema version {schema.get('version', 'unknown')} with {schema.get('num_features', 0)} features")
    return schema


def get_feature_list(schema_path: str = "schema/feature_schema.json") -> List[str]:
    """
    Get the canonical list of feature names in order.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        List of feature names in canonical order
    """
    schema = load_schema(schema_path)
    return schema["features"]


def get_schema_version(schema_path: str = "schema/feature_schema.json") -> str:
    """
    Get the schema version.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Schema version string
    """
    schema = load_schema(schema_path)
    return schema["version"]


def validate_feature_vector(feature_vector: Dict[str, float], 
                           schema_path: str = "schema/feature_schema.json",
                           strict: bool = True) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate that a feature vector matches the schema.
    
    Args:
        feature_vector: Dictionary of feature names to values
        schema_path: Path to schema file
        strict: If True, require all schema features to be present.
                If False, only check that no extra features are present.
        
    Returns:
        Tuple of (is_valid, missing_features)
        - is_valid: True if feature vector matches schema
        - missing_features: List of missing feature names (None if valid)
    """
    schema = load_schema(schema_path)
    required_features = set(schema["features"])
    provided_features = set(feature_vector.keys())
    
    if strict:
        missing = required_features - provided_features
        if missing:
            return False, sorted(list(missing))
    
    extra = provided_features - required_features
    if extra:
        logger.warning(f"Feature vector contains extra features not in schema: {sorted(list(extra))}")
    
    return True, None


def align_feature_vector(feature_vector: Dict[str, float],
                        schema_path: str = "schema/feature_schema.json",
                        fill_missing: float = 0.0) -> List[float]:
    """
    Align a feature vector to the canonical schema order.
    
    This ensures features are in the exact order expected by the model,
    filling missing features with a default value.
    
    Args:
        feature_vector: Dictionary of feature names to values
        schema_path: Path to schema file
        fill_missing: Value to use for missing features
        
    Returns:
        List of feature values in canonical schema order
    """
    schema = load_schema(schema_path)
    feature_list = schema["features"]
    
    aligned = []
    missing = []
    
    for feature_name in feature_list:
        if feature_name in feature_vector:
            aligned.append(feature_vector[feature_name])
        else:
            aligned.append(fill_missing)
            missing.append(feature_name)
    
    if missing:
        logger.warning(f"Missing {len(missing)} features, filled with {fill_missing}: {missing[:10]}...")
    
    return aligned

