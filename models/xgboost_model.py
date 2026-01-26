"""
XGBoost Model for UFC Fight Prediction
Focused, interpretable, and production-ready implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml
import joblib
import json
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from features.feature_pipeline import FeaturePipeline
from database.db_manager import DatabaseManager
from database.schema import Event


class XGBoostModel:
    """XGBoost model for UFC fight prediction with full control and interpretability"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize XGBoost model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.calibrated_model = None
        self.feature_pipeline = FeaturePipeline(config_path, initialize_db=False)
        self.model_dir = Path("models/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_names = None
        self.training_metrics = {}
    
    def create_model(self, 
                     n_estimators: int = 200,
                     max_depth: int = 4,
                     learning_rate: float = 0.05,
                     subsample: float = 0.8,
                     colsample_bytree: float = 0.8,
                     reg_alpha: float = 0.1,
                     reg_lambda: float = 1.0,
                     monotone_constraints: Optional[str] = None,
                     **kwargs) -> xgb.XGBClassifier:
        """
        Create XGBoost classifier with specified parameters
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum depth of trees
            learning_rate: Step size shrinkage
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
        """
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=3, 
            gamma=0.1, 
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )

        # Attach monotonic constraints if provided
        if monotone_constraints is not None:
            params["monotone_constraints"] = monotone_constraints

        params.update(kwargs)

        self.model = xgb.XGBClassifier(**params)

#         model = xgb.XGBClassifier(
#     n_estimators=200,           # More trees = better (to a point)
#     max_depth=4,                # SHALLOW trees = better calibration
#     learning_rate=0.05,         # Slower learning = better generalization
#     subsample=0.8,
#     colsample_bytree=0.8,
#     min_child_weight=3,         # Prevents overfitting
#     gamma=0.1,                  # Regularization
#     reg_alpha=0.1,              # L1 regularization
#     reg_lambda=1.0,             # L2 regularization
#     objective='binary:logistic',
#     eval_metric='logloss',      # CRITICAL: optimize for calibration
#     random_state=42
# )
        
        logger.info(f"Created XGBoost model with parameters:")
        logger.info(f"  n_estimators={n_estimators}, max_depth={max_depth}")
        logger.info(f"  learning_rate={learning_rate}, subsample={subsample}")
        
        return self.model
    
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              early_stopping_rounds: int = 20,
              verbose: bool = True,
              sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Stop if no improvement
            verbose: Print training progress
        
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self.create_model()
        
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Training XGBoost on {len(X_train)} samples...")
        logger.info(f"Features: {len(self.feature_names)}")
        
        # Train model (without early stopping for compatibility)
        if sample_weight is not None:
            logger.info("Using recency-based sample weights for training.")
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        
        # Get training metrics
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_pred = (train_pred_proba > 0.5).astype(int)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_log_loss': log_loss(y_train, train_pred_proba),
            'train_auc': roc_auc_score(y_train, train_pred_proba),
            'n_estimators': self.model.n_estimators,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }
        
        if X_val is not None and y_val is not None:
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_pred = (val_pred_proba > 0.5).astype(int)
            
            metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_log_loss': log_loss(y_val, val_pred_proba),
                'val_auc': roc_auc_score(y_val, val_pred_proba),
                'val_brier': brier_score_loss(y_val, val_pred_proba)
            })
        
        self.training_metrics = metrics
        
        logger.success("Training complete!")
        logger.info(f"Train Accuracy: {metrics['train_accuracy']:.3f}")
        logger.info(f"Train Log Loss: {metrics['train_log_loss']:.4f}")
        logger.info(f"Train AUC: {metrics['train_auc']:.3f}")
        
        if 'val_accuracy' in metrics:
            logger.info(f"Val Accuracy: {metrics['val_accuracy']:.3f}")
            logger.info(f"Val Log Loss: {metrics['val_log_loss']:.4f}")
            logger.info(f"Val AUC: {metrics['val_auc']:.3f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'weight', top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: 'weight', 'gain', or 'cover'
            top_n: Return only top N features
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        importance_df = self.get_feature_importance(top_n=top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            plt.show()
    
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """Plot training learning curves"""
        if not hasattr(self.model, 'evals_result'):
            logger.warning("No evaluation results available. Train with eval_set to see learning curves.")
            return
        
        results = self.model.evals_result()
        
        plt.figure(figsize=(10, 6))
        
        for eval_name, metrics in results.items():
            if 'logloss' in metrics:
                plt.plot(metrics['logloss'], label=f'{eval_name} logloss')
        
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curve to {save_path}")
        else:
            plt.show()
    
    def check_calibration(self, X_test: pd.DataFrame, y_test: pd.Series, 
                         n_bins: int = 10, save_path: Optional[str] = None):
        """
        Check probability calibration
        
        Args:
            X_test: Test features
            y_test: Test labels
            n_bins: Number of bins for calibration curve
            save_path: Path to save plot
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        plt.plot(prob_pred, prob_true, 's-', label='XGBoost', linewidth=2, markersize=8)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Actual Probability', fontsize=12)
        plt.title('Calibration Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration curve to {save_path}")
        else:
            plt.show()
        
        # Calculate calibration metrics
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_test, y_pred_proba)
        
        logger.info(f"Calibration Metrics:")
        logger.info(f"  Brier Score: {brier_score:.4f} (lower is better)")
    
    def calibrate_model(self, X_train: pd.DataFrame, y_train: pd.Series, method: str = 'sigmoid'):
        """
        Calibrate model probabilities
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'sigmoid' or 'isotonic'
        """
        logger.info(f"Calibrating model using {method} method...")
        
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=method,
            cv='prefit'
        )
        
        self.calibrated_model.fit(X_train, y_train)
        logger.success("Model calibrated!")
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             cv: int = 5, n_jobs: int = -1) -> Dict:
        """
        Tune hyperparameters using grid search
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        
        Returns:
            Best parameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            ),
            param_grid,
            cv=cv,
            scoring='neg_log_loss',
            n_jobs=n_jobs,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.success("Hyperparameter tuning complete!")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Accuracy
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Log loss
        logloss_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
        
        # AUC
        auc_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        results = {
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'logloss_mean': -logloss_scores.mean(),
            'logloss_std': logloss_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std()
        }
        
        logger.info(f"Cross-Validation Results:")
        logger.info(f"  Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
        logger.info(f"  Log Loss: {results['logloss_mean']:.4f} ± {results['logloss_std']:.4f}")
        logger.info(f"  AUC: {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
        
        return results
    
    def save_model(self, name: str = "xgboost_model", export_schema: bool = False):
        """
        Save model and metadata
        
        Args:
            name: Model name
            export_schema: If True, also export feature schema to schema/feature_schema.json
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save XGBoost model
        model_path = self.model_dir / f"{name}.json"
        self.model.save_model(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save feature names
        feature_path = self.model_dir / f"{name}_features.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save training metrics
        metrics_path = self.model_dir / f"{name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save feature importance
        importance_df = self.get_feature_importance()
        importance_path = self.model_dir / f"{name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Optionally export feature schema
        if export_schema:
            self.export_feature_schema()
        
        logger.success(f"Model saved to {self.model_dir}")
    
    def load_model(self, name: str = "xgboost_model"):
        """Load saved model"""
        model_path = self.model_dir / f"{name}.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Load feature names
        feature_path = self.model_dir / f"{name}_features.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
        
        logger.success(f"Loaded model from {model_path}")
    
    def export_feature_schema(
        self,
        version: str = "1.0.0",
        output_path: str = "schema/feature_schema.json"
    ) -> Dict:
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
            raise ValueError("No feature names available. Train or load a model first.")
        
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
    
    def predict(self, X: pd.DataFrame, use_calibrated: bool = False) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            use_calibrated: Use calibrated probabilities
        
        Returns:
            Predicted probabilities
        """
        if use_calibrated and self.calibrated_model:
            return self.calibrated_model.predict_proba(X)[:, 1]
        elif self.model:
            return self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No model loaded")


def main():
    """Main function for training XGBoost model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost UFC Fight Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--tune', action='store_true', help='Hyperparameter tuning')
    parser.add_argument('--cross-validate', action='store_true', help='Cross-validation')
    parser.add_argument('--show-importance', action='store_true', help='Show feature importance')
    parser.add_argument('--check-calibration', action='store_true', help='Check calibration')
    parser.add_argument('--plot-learning-curve', action='store_true', help='Plot learning curve')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate probabilities')
    
    # Model parameters
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=6, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='Feature subsample ratio')
    
    # Other options
    parser.add_argument('--data-path', type=str, default='data/processed/training_data.csv', help='Path to training data')
    parser.add_argument('--model-name', type=str, default='xgboost_model', help='Name for saved model (e.g., xgboost_model_with_2025)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--top-n', type=int, default=20, help='Top N features to show')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    parser.add_argument('--export-schema', action='store_true', help='Export feature schema after training')
    parser.add_argument(
        '--holdout-from-year',
        type=int,
        default=None,
        help='Exclude fights with event year >= this from training (e.g. 2025)'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    xgb_model = XGBoostModel()
    
    # Load data
    logger.info("Loading data...")
    pipeline = FeaturePipeline(initialize_db=False)
    df = pipeline.load_dataset(args.data_path)

    # ------------------------------------------------------------------
    # Optional: hold out all fights from a given year (e.g. 2025) from training
    # ------------------------------------------------------------------
    def _split_train_holdout_by_year(raw_df: pd.DataFrame, holdout_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train (events before holdout_year) and holdout (>= holdout_year).
        Uses Event dates from the database via event_id.
        """
        if "event_id" not in raw_df.columns:
            logger.warning("No event_id column in dataset; cannot split by year. Using all rows for training.")
            return raw_df, raw_df.iloc[0:0].copy()

        db = DatabaseManager()
        session = db.get_session()
        try:
            event_ids = (
                raw_df["event_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if not event_ids:
                logger.warning("No event_ids found; using all rows for training.")
                return raw_df, raw_df.iloc[0:0].copy()

            events = (
                session.query(Event)
                .filter(Event.id.in_(event_ids))
                .all()
            )
            id_to_date = {e.id: e.date for e in events}
        finally:
            session.close()

        dates = raw_df["event_id"].map(id_to_date)
        dates_parsed = pd.to_datetime(dates, errors="coerce")
        years = dates_parsed.dt.year

        train_mask = (years < holdout_year) | years.isna()
        holdout_mask = (~train_mask) & years.notna()

        df_train = raw_df[train_mask].copy()
        df_holdout = raw_df[holdout_mask].copy()

        logger.info(
            f"Year-based split with holdout_from_year={holdout_year}: "
            f"{len(df_train)} train rows, {len(df_holdout)} holdout rows."
        )
        return df_train, df_holdout

    if args.holdout_from_year is not None:
        df_train_raw, df_holdout_raw = _split_train_holdout_by_year(df, args.holdout_from_year)
    else:
        df_train_raw, df_holdout_raw = df, df.iloc[0:0].copy()

    # Prepare features only on the training portion
    X, y = pipeline.prepare_features(df_train_raw, fit_scaler=True)

    # ------------------------------------------------------------------
    # Monotonic constraints: load from schema/monotone_constraints.json
    # ------------------------------------------------------------------
    def _load_monotone_constraints(feature_names: List[str]) -> Optional[str]:
        """
        Load monotone constraints from schema/monotone_constraints.json.

        Schema format:
        {
          "version": "1.0.0",
          "num_features": 246,
          "default": 0,
          "constraints": {
            "f1_striking_accuracy": 1,
            "f1_age": -1,
            ...
          }
        }

        Any feature not present in `constraints` uses `default` (usually 0).
        Returns an XGBoost-compatible string: "(0,1,-1,...)",
        or None if the schema file is missing.
        """
        schema_path = Path("schema/monotone_constraints.json")
        if not schema_path.exists():
            logger.warning("No schema/monotone_constraints.json found; training WITHOUT monotone constraints.")
            return None

        with schema_path.open("r") as f:
            data = json.load(f)

        constraints_map = data.get("constraints", {})
        default_val = int(data.get("default", 0))

        constraints: List[int] = []
        for name in feature_names:
            val = constraints_map.get(name, default_val)
            try:
                c = int(val)
            except (TypeError, ValueError):
                c = default_val
            if c not in (-1, 0, 1):
                logger.warning(f"Invalid monotone constraint {c} for feature {name}; using 0 instead.")
                c = 0
            constraints.append(c)

        constraint_str = "(" + ",".join(str(v) for v in constraints) + ")"
        logger.info(f"Using monotone_constraints from schema: {constraint_str}")
        return constraint_str

    monotone_constraints = _load_monotone_constraints(list(X.columns))

    # ------------------------------------------------------------------
    # Recency-based sample weights: newer fights count more than old ones
    # ------------------------------------------------------------------
    def _compute_recency_weights(raw_df: pd.DataFrame, lambda_: float = 0.3) -> pd.Series:
        """
        Compute per-row sample weights based on how long ago the fight happened.

        weight = exp(-lambda * years_ago)
        """
        if "event_id" not in raw_df.columns:
            logger.warning("No event_id column in dataset; using uniform sample weights.")
            return pd.Series(np.ones(len(raw_df)), index=raw_df.index)

        db = DatabaseManager()
        session = db.get_session()
        try:
            event_ids = (
                raw_df["event_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if not event_ids:
                logger.warning("No event_ids found; using uniform sample weights.")
                return pd.Series(np.ones(len(raw_df)), index=raw_df.index)

            events = (
                session.query(Event)
                .filter(Event.id.in_(event_ids))
                .all()
            )
            id_to_date = {e.id: e.date for e in events}
        finally:
            session.close()

        dates = raw_df["event_id"].map(id_to_date)
        dates_parsed = pd.to_datetime(dates, errors="coerce")
        now = pd.Timestamp.now()
        years_ago = (now - dates_parsed).dt.days / 365.25

        weights = np.exp(-lambda_ * years_ago)
        weights.replace([np.inf, -np.inf], np.nan, inplace=True)
        weights.fillna(1.0, inplace=True)

        logger.info(
            f"Recency weights summary — min: {weights.min():.3f}, "
            f"max: {weights.max():.3f}, mean: {weights.mean():.3f}"
        )
        return weights

    recency_weights = _compute_recency_weights(df_train_raw)
    
    # Split data (keep index-based alignment so we can align weights)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    w_train = recency_weights.loc[X_train.index].values
    
    logger.info(f"Data loaded: {len(X_train)} train, {len(X_test)} test")
    
    # Train model
    if args.train or args.evaluate or args.tune:
        if args.tune:
            # Hyperparameter tuning
            best_params = xgb_model.hyperparameter_tuning(X_train, y_train, cv=args.cv_folds)
            xgb_model.train(X_train, y_train, X_test, y_test, sample_weight=w_train)
        else:
            # Regular training with recency weights + monotonic constraints
            xgb_model.create_model(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                monotone_constraints=monotone_constraints,
            )
            xgb_model.train(X_train, y_train, X_test, y_test, sample_weight=w_train)
        
        # Save model (always export feature schema to keep it in sync)
        xgb_model.save_model(name=args.model_name, export_schema=True)
        pipeline.save_pipeline(model_name=args.model_name)
    
    # Cross-validation
    if args.cross_validate:
        xgb_model.cross_validate(X, y, cv=args.cv_folds)
    
    # Feature importance
    if args.show_importance:
        save_path = "models/saved/feature_importance.png" if args.save_plots else None
        xgb_model.plot_feature_importance(top_n=args.top_n, save_path=save_path)
    
    # Learning curve
    if args.plot_learning_curve:
        save_path = "models/saved/learning_curve.png" if args.save_plots else None
        xgb_model.plot_learning_curve(save_path=save_path)
    
    # Calibration
    if args.check_calibration:
        save_path = "models/saved/calibration_curve.png" if args.save_plots else None
        xgb_model.check_calibration(X_test, y_test, save_path=save_path)
    
    # Calibrate model
    if args.calibrate:
        xgb_model.calibrate_model(X_train, y_train)
        xgb_model.save_model(f"{args.model_name}_calibrated", export_schema=True)
    
    logger.success("Done!")


if __name__ == '__main__':
    main()

