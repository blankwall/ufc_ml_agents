"""
Configuration loader for agent_loop.

Loads configuration from agent_loop_config.json and provides
easy access to all configurable parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AgentConfig:
    """Agent execution settings."""
    model: str
    agent_cmd: str
    timeout_seconds: int
    permission_mode: str
    verbose: bool = False
    manual: bool = False


@dataclass(frozen=True)
class ModelPipelineConfig:
    """Model training and feature pipeline settings."""
    holdout_from_year: int
    baseline_json: Path
    train_model_name_prefix: str
    xgboost_predict_model_name: str
    feature_set: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Model evaluation settings."""
    eval_min_year: int
    odds_path: Path
    odds_date_tolerance_days: int
    symmetric: bool


@dataclass(frozen=True)
class IntrospectionConfig:
    """Introspection and analysis settings."""
    non_feature_columns: List[str]
    low_importance_threshold: float
    top_n_features: int
    concentration_thresholds: Dict[str, int]
    feature_count_thresholds: Dict[str, int]


@dataclass(frozen=True)
class PathsConfig:
    """Common paths used throughout the agent loop."""
    repo_root: Path
    agent_artifacts_dir: Path
    data_processed_dir: Path
    schema_dir: Path
    features_dir: Path
    models_saved_dir: Path
    reports_dir: Path
    prompts_dir: Path


@dataclass(frozen=True)
class FeatureSchemaConfig:
    """Feature schema related paths."""
    feature_schema_path: Path
    monotone_constraints_path: Path
    feature_exclusions_path: Path


@dataclass(frozen=True)
class FeatureTypesConfig:
    """Patterns for identifying feature types."""
    interaction_pattern: str
    ratio_patterns: List[str]
    polynomial_patterns: List[str]
    aggregate_patterns: List[str]


@dataclass(frozen=True)
class AgentLoopConfig:
    """Main configuration container for agent_loop."""
    agent: AgentConfig
    model_pipeline: ModelPipelineConfig
    evaluation: EvaluationConfig
    introspection: IntrospectionConfig
    paths: PathsConfig
    feature_schema: FeatureSchemaConfig
    feature_types: FeatureTypesConfig
    roi_thresholds: Dict[str, float]
    accuracy_thresholds: Dict[str, float]

    @classmethod
    def from_json(cls, config_path: Path, repo_root: Optional[Path] = None) -> "AgentLoopConfig":
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to config JSON file
            repo_root: Repository root path (overrides config or uses cwd)

        Returns:
            AgentLoopConfig instance
        """
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Resolve repo_root
        if repo_root is None:
            repo_root = Path(data.get("paths", {}).get("repo_root", "."))
        repo_root = repo_root.resolve()

        # Build nested config objects
        agent_cfg = AgentConfig(**data["agent"])

        pipeline_data = data["model_pipeline"].copy()
        pipeline_data["baseline_json"] = repo_root / pipeline_data["baseline_json"]
        pipeline_cfg = ModelPipelineConfig(**pipeline_data)

        eval_data = data["evaluation"].copy()
        eval_data["odds_path"] = repo_root / eval_data["odds_path"]
        eval_cfg = EvaluationConfig(**eval_data)

        introspection_cfg = IntrospectionConfig(**data["introspection"])

        paths_data = data["paths"].copy()
        paths_data["repo_root"] = repo_root
        paths_cfg = PathsConfig(
            **{k: repo_root / v if k != "repo_root" else v for k, v in paths_data.items()}
        )

        schema_data = data["feature_schema"].copy()
        schema_cfg = FeatureSchemaConfig(
            **{k: repo_root / v for k, v in schema_data.items()}
        )

        feature_types_cfg = FeatureTypesConfig(**data["feature_types"])

        roi_thresholds = data["roi_thresholds"]
        accuracy_thresholds = data["accuracy_thresholds"]

        return cls(
            agent=agent_cfg,
            model_pipeline=pipeline_cfg,
            evaluation=eval_cfg,
            introspection=introspection_cfg,
            paths=paths_cfg,
            feature_schema=schema_cfg,
            feature_types=feature_types_cfg,
            roi_thresholds=roi_thresholds,
            accuracy_thresholds=accuracy_thresholds,
        )

    @classmethod
    def with_overrides(cls, base_config: "AgentLoopConfig", **overrides) -> "AgentLoopConfig":
        """
        Create a new config with specific overrides.

        Args:
            base_config: Base configuration to copy
            **overrides: Field values to override (use dot notation, e.g., agent.model="new-model")

        Returns:
            New AgentLoopConfig with overrides applied
        """
        # For now, return a copy of the base config
        # A more sophisticated implementation would apply overrides
        return base_config


def load_config(
    config_path: Optional[Path] = None,
    repo_root: Optional[Path] = None
) -> AgentLoopConfig:
    """
    Load agent_loop configuration.

    Args:
        config_path: Path to config file (default: agent_loop/agent_loop_config.json)
        repo_root: Repository root path

    Returns:
        AgentLoopConfig instance
    """
    if config_path is None:
        # Default to agent_loop_config.json in the same directory as this file
        config_path = Path(__file__).parent / "agent_loop_config.json"

    return AgentLoopConfig.from_json(config_path, repo_root)
