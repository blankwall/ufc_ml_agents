"""
Feature Pipeline - Orchestrates feature extraction and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from database.db_manager import DatabaseManager
from .matchup_features import create_training_dataset
from .registry import FeatureRegistry

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class FeaturePipeline:
    """Manages the complete feature engineering pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml", initialize_db: bool = True):
        """
        Initialize feature pipeline
        
        Args:
            config_path: Path to YAML config
            initialize_db: Whether to create a database connection (needed for
                dataset creation, but can be disabled for fast inference)
        """
        self.db = DatabaseManager(config_path) if initialize_db else None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_dataset(
        self,
        output_path: str = 'data/processed/training_data.csv',
        feature_set: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Create complete training dataset

        Args:
            output_path: Path to save the dataset
            feature_set: Optional list of feature names to extract.
                        If None, uses FEATURE_SET_FULL (all features)
            show_progress: Show progress bar if tqdm is available (default: True)
        """
        if self.db is None:
            raise RuntimeError("FeaturePipeline was initialized with initialize_db=False, "
                               "cannot create dataset without a database connection.")
        session = self.db.get_session()
        try:
            df = create_training_dataset(session, output_path, feature_set=feature_set, show_progress=show_progress)
            return df
        finally:
            session.close()
    
    def load_dataset(self, file_path: str = 'data/processed/training_data.csv', 
                    batch_mode: bool = True, show_progress: bool = True) -> pd.DataFrame:
        """
        Load preprocessed dataset with batch loading support.

        Args:
            file_path: Path to the CSV file
            batch_mode: Use batch loading mode (much faster! default: True)
            show_progress: Show progress bar if tqdm is available (default: True)

        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading dataset from {file_path}")

        if batch_mode:
            logger.info("Using batch loading mode (much faster!)")
            # Show progress bar if available and requested
            if show_progress and TQDM_AVAILABLE:
                logger.info("Progress bar enabled (tqdm available)")
            elif show_progress and not TQDM_AVAILABLE:
                logger.info("Install tqdm for progress bar: pip install tqdm")

        df = pd.read_csv(file_path)
        logger.success(f"Loaded {len(df)} samples with {len(df.columns)} features")

        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        fit_scaler: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        
        Args:
            df: Raw dataframe
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Preparing features for training...")
        
        # Separate features and target
        metadata_cols = ['fight_id', 'event_id', 'fighter_1_id', 'fighter_2_id', 
                        'weight_class', 'method', 'target']
        
        # Deterministic feature ordering:
        # Pandas column order can vary depending on how the dataset was assembled.
        # Sorting here makes training/inference reproducible (and schema export stable).
        feature_cols = sorted([col for col in df.columns if col not in metadata_cols])
        
        X = df[feature_cols].copy()
        y = df['target'] if 'target' in df.columns else None
        
        # For inference, align columns to the feature set seen during training.
        # This avoids XGBoost "feature names should match" errors by:
        #   - adding any missing training-time features with 0
        #   - dropping any new columns not seen during training
        if not fit_scaler and self.feature_names is not None:
            # Add missing expected columns
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                for c in missing:
                    X[c] = 0.0
            # Drop unexpected columns and order to match training
            X = X[self.feature_names]
        else:
            # During training, remember the column order
            self.feature_names = list(X.columns)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted scaler on training data")
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.success(f"Prepared {len(feature_cols)} features")
        
        return X_scaled, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_pipeline(self, output_dir: str = 'models/saved', model_name: Optional[str] = None):
        """
        Save the feature pipeline (scaler + feature names).

        IMPORTANT:
        - If you train multiple models (e.g. `xgboost_model` and `xgboost_model_with_2025`)
          you must save the pipeline artifacts per-model, otherwise the last training run
          overwrites `feature_names.pkl`/`feature_scaler.pkl` and breaks older models with
          XGBoost "feature_names mismatch" errors.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if model_name:
            scaler_path = output_path / f"{model_name}_feature_scaler.pkl"
            features_path = output_path / f"{model_name}_feature_names.pkl"
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_names, features_path)
            logger.success(f"Saved feature pipeline to {output_dir} (model_name={model_name})")
        else:
            # Legacy single-model paths (kept for backward compatibility)
            scaler_path = output_path / 'feature_scaler.pkl'
            features_path = output_path / 'feature_names.pkl'
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_names, features_path)
            logger.success(f"Saved feature pipeline to {output_dir}")
    
    def load_pipeline(self, input_dir: str = 'models/saved', model_name: Optional[str] = None):
        """
        Load a saved feature pipeline.

        If `model_name` is provided, we will try to load:
          - `{model_name}_feature_scaler.pkl`
          - `{model_name}_feature_names.pkl`
        and fall back to the legacy single-model filenames if those don't exist.
        """
        input_path = Path(input_dir)

        def _load_paths(scaler_path: Path, features_path: Path) -> None:
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)

        if model_name:
            scaler_path = input_path / f"{model_name}_feature_scaler.pkl"
            features_path = input_path / f"{model_name}_feature_names.pkl"
            if scaler_path.exists() and features_path.exists():
                _load_paths(scaler_path, features_path)
                logger.success(f"Loaded feature pipeline from {input_dir} (model_name={model_name})")
            else:
                # Fallback to legacy paths
                legacy_scaler = input_path / "feature_scaler.pkl"
                legacy_features = input_path / "feature_names.pkl"
                _load_paths(legacy_scaler, legacy_features)
                logger.warning(
                    f"Model-specific pipeline files not found for '{model_name}'. "
                    f"Fell back to legacy pipeline files in {input_dir}. "
                    f"If you see feature mismatch errors, retrain and ensure the pipeline "
                    f"is saved with model_name='{model_name}'."
                )
        else:
            legacy_scaler = input_path / "feature_scaler.pkl"
            legacy_features = input_path / "feature_names.pkl"
            _load_paths(legacy_scaler, legacy_features)
            logger.success(f"Loaded feature pipeline from {input_dir}")

        # Optional integrity check: if a model-specific feature list exists, ensure alignment.
        # This catches the common pitfall where a different model overwrote the legacy pipeline files.
        if model_name:
            model_features_path = Path("models/saved") / f"{model_name}_features.json"
            if model_features_path.exists():
                with model_features_path.open("r") as f:
                    model_features = json.load(f)
                if isinstance(model_features, list) and self.feature_names is not None:
                    if list(model_features) != list(self.feature_names):
                        raise ValueError(
                            f"Loaded pipeline feature_names do not match model '{model_name}' feature list. "
                            f"Pipeline features: {len(self.feature_names)}; model features: {len(model_features)}. "
                            f"This usually means a different model overwrote legacy pipeline files. "
                            f"Fix: retrain '{model_name}' (or re-save its pipeline) so that "
                            f"models/saved/{model_name}_feature_names.pkl and "
                            f"models/saved/{model_name}_feature_scaler.pkl exist."
                        )
    
    def get_feature_importance_summary(self, feature_importances: np.ndarray, 
                                       top_n: int = 20) -> pd.DataFrame:
        """
        Create a summary of feature importances
        
        Args:
            feature_importances: Array of feature importance scores
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_names is None:
            logger.warning("Feature names not set")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} most important features:")
        logger.info("\n" + importance_df.head(top_n).to_string())
        
        return importance_df
    
    def export_feature_schema(
        self,
        version: str = "1.0.0",
        output_path: str = "schema/feature_schema.json"
    ) -> dict:
        """
        Export feature schema to JSON file.
        
        This creates the canonical feature schema that becomes the master contract
        between training, prediction, Excel export, and API usage.
        
        Args:
            version: Schema version string
            output_path: Path to save the schema file
            
        Returns:
            Schema dictionary
        """
        if self.feature_names is None:
            raise ValueError("No feature names available. Prepare features first.")
        
        import json
        
        schema = {
            "version": version,
            "num_features": len(self.feature_names),
            "features": self.feature_names
        }
        
        # Save to file
        schema_path = Path(output_path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.success(f"Exported feature schema to {schema_path}")
        logger.info(f"Schema version: {version}")
        logger.info(f"Total features: {len(self.feature_names)}")
        
        return schema


def main():
    """Main function for running the feature pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description='UFC Feature Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m features.feature_pipeline --create
  python -m features.feature_pipeline --create --feature-set advanced
  python -m features.feature_pipeline --prepare --batch-mode
  python -m features.feature_pipeline --prepare --no-progress
        """
    )
    parser.add_argument('--create', action='store_true',
                       help='Create training dataset from database')
    parser.add_argument('--prepare', action='store_true',
                       help='Prepare features for training')
    parser.add_argument('--input', type=str,
                       default='data/processed/training_data.csv',
                       help='Input dataset path (default: data/processed/training_data.csv)')
    parser.add_argument('--output', type=str,
                       default='data/processed/prepared_data.csv',
                       help='Output dataset path (default: data/processed/prepared_data.csv)')
    parser.add_argument('--feature-set', type=str,
                       choices=['base', 'advanced', 'full'],
                       default='full',
                       help='Feature set to use: base, advanced, or full (default: full)')
    parser.add_argument('--export-schema', action='store_true',
                       help='Export feature schema after preparing features (requires --prepare)')
    parser.add_argument('--schema-version', type=str, default='1.0.0',
                       help='Schema version string (default: 1.0.0)')
    parser.add_argument('--batch-mode', action='store_true', default=True,
                       help='Use batch loading mode (much faster! default: True)')
    parser.add_argument('--no-batch-mode', dest='batch_mode', action='store_false',
                       help='Disable batch loading mode')
    parser.add_argument('--progress', action='store_true', default=True,
                       help='Show progress bar if tqdm is installed (default: True)')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                       help='Disable progress bar (applies to both --create and --prepare)')

    args = parser.parse_args()

    pipeline = FeaturePipeline()

    if args.create:
        # Map feature set string to actual feature set
        feature_set_map = {
            'base': FeatureRegistry.FEATURE_SET_BASE,
            'advanced': FeatureRegistry.FEATURE_SET_ADVANCED,
            'full': FeatureRegistry.FEATURE_SET_FULL,
        }
        feature_set = feature_set_map[args.feature_set]

        logger.info(f"Creating training dataset with '{args.feature_set}' feature set...")
        df = pipeline.create_dataset(feature_set=feature_set, show_progress=args.progress)
        logger.info(f"Created dataset with shape: {df.shape}")
        logger.info(f"Total features: {len([c for c in df.columns if c not in ['fight_id', 'event_id', 'fighter_1_id', 'fighter_2_id', 'weight_class', 'method', 'target', 'is_title_fight']])}")

    if args.prepare:
        df = pipeline.load_dataset(args.input, batch_mode=args.batch_mode, show_progress=args.progress)
        X, y = pipeline.prepare_features(df, fit_scaler=True)

        # Save prepared data
        prepared_df = X.copy()
        prepared_df['target'] = y
        prepared_df.to_csv(args.output, index=False)

        # Save pipeline
        pipeline.save_pipeline()

        logger.success(f"Prepared data saved to {args.output}")

        # Optionally export feature schema
        if args.export_schema:
            pipeline.export_feature_schema(version=args.schema_version)


if __name__ == '__main__':
    main()

