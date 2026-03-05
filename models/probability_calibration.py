"""
Probability Calibration Module

Fixes overconfidence in the 50-70% probability range that's causing
favorites to lose money (-25% ROI).

Calibration strategy:
1. Analyze validation data to find calibration curves
2. Apply piecewise linear calibration to correct overconfident ranges
3. Preserve underdog performance while fixing favorites
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


class ProbabilityCalibrator:
    """
    Calibrates model probabilities to fix overconfidence/underconfidence issues.

    The current model shows:
    - 50-60% prob bucket: 48.3% actual (overconfident by ~4%)
    - 60-70% prob bucket: 53.3% actual (overconfident by ~12%)
    - 70-80% prob bucket: 73.3% actual (well calibrated)
    - >80% prob bucket: 88.9% actual (well calibrated)
    """

    def __init__(self, method: str = 'piecewise_linear'):
        """
        Initialize calibrator.

        Args:
            method: Calibration method
                - 'piecewise_linear': Simple piecewise linear calibration (fast, interpretable)
                - 'isotonic': Isotonic regression (more flexible, requires more data)
                - 'platt': Platt scaling (logistic regression)
        """
        self.method = method
        self.is_fitted = False
        self.calibration_params = {}

        # For piecewise linear calibration
        self.breakpoints = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
        # These will be learned from data
        self.calibrated_outputs = [0.0, 0.5, 0.53, 0.58, 0.73, 1.0]

        # For isotonic regression
        self.isotonic_regressor = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation data.

        Args:
            y_true: True labels (0 or 1)
            y_prob: Predicted probabilities from model

        Returns:
            self (fitted calibrator)
        """
        logger.info(f"Fitting probability calibrator using method: {self.method}")

        # Calculate calibration curve
        self._analyze_calibration(y_true, y_prob)

        if self.method == 'piecewise_linear':
            self._fit_piecewise_linear(y_true, y_prob)
        elif self.method == 'isotonic':
            self._fit_isotonic(y_true, y_prob)
        elif self.method == 'platt':
            self._fit_platt(y_true, y_prob)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        logger.success("Calibrator fitted successfully")

        return self

    def _analyze_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Analyze calibration across probability buckets."""
        df = pd.DataFrame({'true': y_true, 'prob': y_prob})

        # Create probability buckets
        buckets = [
            (0.0, 0.4, '<40%'),
            (0.4, 0.5, '40-50%'),
            (0.5, 0.6, '50-60%'),
            (0.6, 0.7, '60-70%'),
            (0.7, 0.8, '70-80%'),
            (0.8, 1.0, '>80%'),
        ]

        logger.info("Calibration analysis:")
        for low, high, label in buckets:
            mask = (df['prob'] >= low) & (df['prob'] < high)
            bucket_data = df[mask]
            if len(bucket_data) > 0:
                actual_rate = bucket_data['true'].mean()
                avg_prob = bucket_data['prob'].mean()
                error = actual_rate - avg_prob
                logger.info(f"  {label}: avg_prob={avg_prob:.3f}, actual={actual_rate:.3f}, error={error:+.3f}, n={len(bucket_data)}")

    def _fit_piecewise_linear(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit piecewise linear calibration.

        This is our main approach - it's interpretable and targets specific problem ranges.
        """
        df = pd.DataFrame({'true': y_true, 'prob': y_prob})

        # Calculate actual win rates for each bucket
        buckets = [
            (0.0, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),  # Problem range 1
            (0.6, 0.7),  # Problem range 2
            (0.7, 0.8),
            (0.8, 1.0),
        ]

        calibration_points = []

        for low, high in buckets:
            mask = (df['prob'] >= low) & (df['prob'] < high)
            bucket_data = df[mask]

            if len(bucket_data) >= 10:  # Need minimum samples
                # Use midpoint of bucket as input, actual win rate as output
                midpoint = (low + high) / 2
                actual_rate = bucket_data['true'].mean()

                # Smooth calibration: blend with identity (no correction)
                # This prevents overfitting to small samples
                blend_factor = 0.7  # 70% actual rate, 30% identity
                calibrated = midpoint * (1 - blend_factor) + actual_rate * blend_factor

                calibration_points.append((midpoint, calibrated))
            else:
                # Not enough data, use identity (no calibration)
                calibration_points.append(((low + high) / 2, (low + high) / 2))

        # Store breakpoints and calibrated outputs
        self.breakpoints = [0.0] + [p[0] for p in calibration_points] + [1.0]
        self.calibrated_outputs = [0.0] + [p[1] for p in calibration_points] + [1.0]

        # Store parameters
        self.calibration_params['breakpoints'] = self.breakpoints
        self.calibration_params['outputs'] = self.calibrated_outputs

        logger.info(f"Piecewise linear calibration: {len(self.breakpoints)} breakpoints")

    def _fit_isotonic(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit isotonic regression calibration."""
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(y_prob, y_true)
        logger.info("Isotonic regression fitted")

    def _fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression)."""
        # Platt scaling fits: logit(p) = a * z + b
        # where z is the model logit and p is the calibrated probability
        # We implement this directly using simple optimization

        from scipy.optimize import minimize

        def platt_loss(params):
            """Negative log likelihood for Platt scaling."""
            a, b = params
            # Apply Platt scaling
            z = np.log(y_prob / (1 - y_prob + 1e-15))
            p_calib = 1 / (1 + np.exp(-(a * z + b)))

            # Avoid log(0)
            p_calib = np.clip(p_calib, 1e-15, 1 - 1e-15)

            # Binary cross-entropy
            loss = -(y_true * np.log(p_calib) + (1 - y_true) * np.log(1 - p_calib))
            return loss.mean()

        # Initialize with identity (a=1, b=0)
        result = minimize(platt_loss, x0=[1.0, 0.0], method='Nelder-Mead')

        self.calibration_params['platt_a'] = result.x[0]
        self.calibration_params['platt_b'] = result.x[1]
        logger.info(f"Platt scaling: a={result.x[0]:.3f}, b={result.x[1]:.3f}")

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            y_prob: Raw model probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return y_prob

        y_prob = np.asarray(y_prob)
        y_prob_clipped = np.clip(y_prob, 0.0, 1.0)

        if self.method == 'piecewise_linear':
            return self._calibrate_piecewise_linear(y_prob_clipped)
        elif self.method == 'isotonic':
            return self.isotonic_regressor.predict(y_prob_clipped)
        elif self.method == 'platt':
            return self._calibrate_platt(y_prob_clipped)
        else:
            return y_prob

    def _calibrate_piecewise_linear(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply piecewise linear calibration."""
        calibrated = np.zeros_like(y_prob)

        for i, prob in enumerate(y_prob):
            # Find which segment this probability falls into
            for j in range(len(self.breakpoints) - 1):
                if self.breakpoints[j] <= prob <= self.breakpoints[j + 1]:
                    # Linear interpolation between calibration points
                    x0, x1 = self.breakpoints[j], self.breakpoints[j + 1]
                    y0, y1 = self.calibrated_outputs[j], self.calibrated_outputs[j + 1]

                    if x1 != x0:
                        calibrated[i] = y0 + (y1 - y0) * (prob - x0) / (x1 - x0)
                    else:
                        calibrated[i] = y0
                    break
            else:
                # Outside range, use nearest endpoint
                calibrated[i] = self.calibrated_outputs[-1] if prob > self.breakpoints[-2] else self.calibrated_outputs[0]

        # Clip to valid probability range
        return np.clip(calibrated, 0.0, 1.0)

    def _calibrate_platt(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        a = self.calibration_params['platt_a']
        b = self.calibration_params['platt_b']

        z = np.log(y_prob / (1 - y_prob + 1e-15))
        z_calib = a * z + b
        p_calib = 1 / (1 + np.exp(-z_calib))

        return np.clip(p_calib, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        import json

        data = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'calibration_params': self.calibration_params,
        }

        if self.method == 'piecewise_linear':
            data['breakpoints'] = self.breakpoints
            data['calibrated_outputs'] = self.calibrated_outputs

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.success(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        """Load calibrator from disk."""
        import pickle

        with open(path, 'rb') as f:
            calibrator = pickle.load(f)

        logger.success(f"Calibrator loaded from {path}")
        return calibrator


def create_favorites_focused_calibrator() -> ProbabilityCalibrator:
    """
    Create a calibrator specifically tuned to fix the favorites problem.

    Based on validation data analysis:
    - Favorites lose money in 50-70% probability range
    - Need to dampen overconfidence by 4-12% in this range

    Returns:
        Pre-configured calibrator with targeted calibration points
    """
    calibrator = ProbabilityCalibrator(method='piecewise_linear')

    # Pre-set calibration based on validation analysis
    # These values target the overconfidence we observed:
    calibrator.breakpoints = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    # Target calibration: dampen 50-70% range
    # - 50% actual should be ~50% (no change needed at 50%)
    # - 55% predicted should be calibrated to ~51% (4% reduction)
    # - 60% predicted should be calibrated to ~53% (7% reduction)
    # - 65% predicted should be calibrated to ~58% (7% reduction)
    # - 70% predicted should be calibrated to ~66% (4% reduction)
    calibrator.calibrated_outputs = [0.0, 0.40, 0.48, 0.51, 0.55, 0.70, 1.0]

    calibrator.is_fitted = True
    calibrator.calibration_params = {
        'breakpoints': calibrator.breakpoints,
        'outputs': calibrator.calibrated_outputs,
    }

    logger.info("Created favorites-focused calibrator with pre-set calibration points")
    logger.info(f"Calibration map: {list(zip(calibrator.breakpoints, calibrator.calibrated_outputs))}")

    return calibrator


# Convenience function for quick calibration
def calibrate_predictions(y_prob: np.ndarray,
                          method: str = 'favorites_fix') -> np.ndarray:
    """
    Quick calibration without needing to fit first.

    Args:
        y_prob: Raw model probabilities
        method: Calibration method ('favorites_fix' uses pre-tuned values)

    Returns:
        Calibrated probabilities
    """
    if method == 'favorites_fix':
        calibrator = create_favorites_focused_calibrator()
        return calibrator.calibrate(y_prob)
    else:
        return y_prob


if __name__ == '__main__':
    # Test the calibrator
    print("Testing Probability Calibrator")
    print("=" * 60)

    # Test piecewise linear calibration
    calibrator = create_favorites_focused_calibrator()

    # Test probabilities
    test_probs = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85])
    calibrated = calibrator.calibrate(test_probs)

    print("\nCalibration test:")
    for raw, cal in zip(test_probs, calibrated):
        diff = cal - raw
        print(f"  {raw:.2f} -> {cal:.2f} ({diff:+.2f})")
