"""
ML Models for UFC fight prediction
"""

from typing import TYPE_CHECKING

# IMPORTANT:
# Avoid importing heavyweight optional dependencies (e.g. torch) at package import time.
# Many scripts only need `models.xgboost_model` and should not be forced to import
# the ensemble / neural net stack.
#
# If you want to expose these at the package level, use lazy imports in the caller
# or import directly from the module (e.g. `from models.ensemble import EnsembleModel`).
if TYPE_CHECKING:
    from .baseline_models import BaselineModels  # noqa: F401
    from .ensemble import EnsembleModel  # noqa: F401

__all__ = []

