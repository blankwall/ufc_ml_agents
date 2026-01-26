"""
Generate Feature Schema from Trained Model

This script extracts the feature names from a trained model and saves them
as the canonical feature schema. This schema becomes the master contract
between training, prediction, Excel export, and API usage.
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional
from loguru import logger

from models.xgboost_model import XGBoostModel
from features.feature_pipeline import FeaturePipeline


def load_feature_names_from_model(model_name: str = "xgboost_model") -> Optional[List[str]]:
    """
    Load feature names from a saved XGBoost model.
    
    Args:
        model_name: Name of the saved model
        
    Returns:
        List of feature names or None if not found
    """
    model_dir = Path("models/saved")
    feature_path = model_dir / f"{model_name}_features.json"
    
    if not feature_path.exists():
        logger.error(f"Feature file not found: {feature_path}")
        return None
    
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    logger.info(f"Loaded {len(feature_names)} features from {feature_path}")
    return feature_names


def load_feature_names_from_pipeline() -> Optional[List[str]]:
    """
    Load feature names from a saved feature pipeline.
    
    Returns:
        List of feature names or None if not found
    """
    import joblib
    
    pipeline_dir = Path("models/saved")
    feature_path = pipeline_dir / "feature_names.pkl"
    
    if not feature_path.exists():
        logger.error(f"Feature file not found: {feature_path}")
        return None
    
    feature_names = joblib.load(feature_path)
    logger.info(f"Loaded {len(feature_names)} features from {feature_path}")
    return feature_names


def load_feature_names_from_xgboost_booster(model_name: str = "xgboost_model") -> Optional[List[str]]:
    """
    Load feature names directly from XGBoost booster.
    
    Args:
        model_name: Name of the saved model
        
    Returns:
        List of feature names or None if not found
    """
    import xgboost as xgb
    
    model_dir = Path("models/saved")
    model_path = model_dir / f"{model_name}.json"
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    feature_names = model.get_booster().feature_names
    logger.info(f"Loaded {len(feature_names)} features from XGBoost booster")
    return feature_names


def generate_schema(
    feature_names: List[str],
    version: str = "1.0.0",
    output_path: str = "schema/feature_schema.json"
) -> dict:
    """
    Generate feature schema dictionary.
    
    Args:
        feature_names: List of feature names in canonical order
        version: Schema version string
        output_path: Path to save the schema
        
    Returns:
        Schema dictionary
    """
    schema = {
        "version": version,
        "features": feature_names,
        "num_features": len(feature_names)
    }
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.success(f"Saved feature schema to {output_file}")
    logger.info(f"Schema version: {version}")
    logger.info(f"Total features: {len(feature_names)}")
    
    return schema


def load_schema(schema_path: str = "schema/feature_schema.json") -> dict:
    """
    Load feature schema from file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Schema dictionary
    """
    schema_file = Path(schema_path)
    
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    
    logger.info(f"Loaded schema version {schema.get('version', 'unknown')} with {schema.get('num_features', 0)} features")
    return schema


def main():
    """Main function for generating feature schema"""
    parser = argparse.ArgumentParser(
        description='Generate feature schema from trained model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='xgboost_model',
        help='Name of the saved model (default: xgboost_model)'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['model', 'pipeline', 'booster'],
        default='model',
        help='Source to load features from: model (JSON), pipeline (PKL), or booster (XGBoost)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='1.0.0',
        help='Schema version (default: 1.0.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='schema/feature_schema.json',
        help='Output path for schema file (default: schema/feature_schema.json)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the generated schema matches the model'
    )
    
    args = parser.parse_args()
    
    # Load feature names based on source
    if args.source == 'model':
        feature_names = load_feature_names_from_model(args.model_name)
    elif args.source == 'pipeline':
        feature_names = load_feature_names_from_pipeline()
    elif args.source == 'booster':
        feature_names = load_feature_names_from_xgboost_booster(args.model_name)
    else:
        logger.error(f"Unknown source: {args.source}")
        return
    
    if feature_names is None:
        logger.error("Failed to load feature names")
        return
    
    # Generate and save schema
    schema = generate_schema(feature_names, version=args.version, output_path=args.output)
    
    # Verify if requested
    if args.verify:
        logger.info("Verifying schema...")
        
        # Try loading from different sources and compare
        if args.source != 'booster':
            booster_features = load_feature_names_from_xgboost_booster(args.model_name)
            if booster_features:
                if booster_features == feature_names:
                    logger.success("Schema matches XGBoost booster feature names")
                else:
                    logger.warning("Schema does not match XGBoost booster feature names")
                    logger.warning(f"Schema has {len(feature_names)} features, booster has {len(booster_features)}")
        
        if args.source != 'pipeline':
            pipeline_features = load_feature_names_from_pipeline()
            if pipeline_features:
                if pipeline_features == feature_names:
                    logger.success("Schema matches pipeline feature names")
                else:
                    logger.warning("Schema does not match pipeline feature names")
                    logger.warning(f"Schema has {len(feature_names)} features, pipeline has {len(pipeline_features)}")


if __name__ == '__main__':
    main()

