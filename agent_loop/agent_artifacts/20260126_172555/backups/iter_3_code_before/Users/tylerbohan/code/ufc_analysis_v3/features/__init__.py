"""
Feature engineering module for UFC fight prediction

REFACTORED: Modular feature system with clean separation of concerns.
"""

# Main interfaces (backward compatible)
from .fighter_features import FighterFeatureExtractor
from .matchup_features import MatchupFeatureExtractor
from .feature_pipeline import FeaturePipeline

# New modular system
from .registry import FeatureBuilder, FeatureRegistry
from .feature_vector_builder import (
    build_feature_vector,
    build_feature_vector_from_matchup,
    build_feature_vectors_batch,
    get_schema_info,
)

# Feature modules (for direct use if needed)
from . import physical
from . import striking
from . import grappling
from . import experiential
from . import time_based
from . import opponent_quality
from . import utils

__all__ = [
    # Main interfaces
    'FighterFeatureExtractor',
    'MatchupFeatureExtractor',
    'FeaturePipeline',
    # New modular system
    'FeatureBuilder',
    'FeatureRegistry',
    # Feature vector builder (schema-enforced)
    'build_feature_vector',
    'build_feature_vector_from_matchup',
    'build_feature_vectors_batch',
    'get_schema_info',
    # Feature modules
    'physical',
    'striking',
    'grappling',
    'experiential',
    'time_based',
    'opponent_quality',
    'utils',
]

