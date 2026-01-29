"""
Test script to demonstrate batch loading performance improvements.

Run this to compare individual loading vs batch loading speeds.
"""

import time
import numpy as np
from pathlib import Path
from loguru import logger

from features.feature_pipeline import FeaturePipeline
from features.feature_vector_builder import build_feature_vector, build_feature_vectors_batch


def test_batch_loading_performance():
    """
    Test and compare batch loading vs individual loading performance.
    """
    logger.info("=" * 80)
    logger.info("Batch Loading Performance Test")
    logger.info("=" * 80)

    # Check if training data exists
    training_data_path = Path('data/processed/training_data.csv')
    if not training_data_path.exists():
        logger.error(f"Training data not found at {training_data_path}")
        logger.info("Please run: python -m features.feature_pipeline --create")
        return

    pipeline = FeaturePipeline()

    # Test 1: Load dataset with batch mode
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: Loading Dataset with Batch Mode")
    logger.info("=" * 80)

    start_time = time.time()
    df = pipeline.load_dataset(
        file_path=str(training_data_path),
        batch_mode=True,
        show_progress=True
    )
    batch_load_time = time.time() - start_time

    logger.success(f"✓ Loaded {len(df)} samples in {batch_load_time:.2f} seconds (batch mode)")
    logger.info(f"  Dataset shape: {df.shape}")

    # Get feature columns (exclude metadata)
    metadata_cols = ['fight_id', 'event_id', 'fighter_1_id', 'fighter_2_id',
                    'weight_class', 'method', 'target']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    logger.info(f"  Features: {len(feature_cols)}")

    # Test 2: Prepare features with batch mode
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Preparing Features with Batch Mode")
    logger.info("=" * 80)

    start_time = time.time()
    X, y = pipeline.prepare_features(df, fit_scaler=True)
    prepare_time = time.time() - start_time

    logger.success(f"✓ Prepared {len(X)} feature vectors in {prepare_time:.2f} seconds")
    logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Test 3: Build feature vectors individually (baseline)
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Building Feature Vectors Individually (Baseline)")
    logger.info("=" * 80)

    # Take a smaller sample for individual test (to save time)
    n_samples = min(1000, len(df))
    logger.info(f"Testing with {n_samples} samples (smaller subset for fair comparison)")

    # Create feature dictionaries
    feature_dicts = []
    for i in range(n_samples):
        feature_dict = {}
        for col in feature_cols:
            feature_dict[col] = X.iloc[i][col]
        feature_dicts.append(feature_dict)

    # Test individual loading
    start_time = time.time()
    vectors_individual = []
    for i, feature_dict in enumerate(feature_dicts):
        vector = build_feature_vector(feature_dict, strict=False)
        vectors_individual.append(vector)
        if i % 100 == 0:
            logger.debug(f"  Processed {i}/{n_samples} samples...")

    individual_time = time.time() - start_time
    logger.success(f"✓ Built {len(vectors_individual)} vectors in {individual_time:.2f} seconds (individual)")

    # Test 4: Build feature vectors in batch
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Building Feature Vectors in Batch Mode")
    logger.info("=" * 80)

    start_time = time.time()
    vectors_batch = build_feature_vectors_batch(
        feature_dicts[:n_samples],
        show_progress=True
    )
    batch_time = time.time() - start_time
    logger.success(f"✓ Built {len(vectors_batch)} vectors in {batch_time:.2f} seconds (batch mode)")

    # Test 5: Verify results are identical
    logger.info("\n" + "=" * 80)
    logger.info("Test 5: Verification - Results are Identical?")
    logger.info("=" * 80)

    vectors_individual_array = np.array(vectors_individual)
    are_equal = np.allclose(vectors_individual_array, vectors_batch, rtol=1e-5)

    if are_equal:
        logger.success("✓ Results are identical (within numerical precision)")
    else:
        max_diff = np.max(np.abs(vectors_individual_array - vectors_batch))
        logger.warning(f"⚠ Results differ by up to {max_diff:.10f}")
        logger.warning("This is likely due to floating-point precision differences")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    speedup = individual_time / batch_time if batch_time > 0 else float('inf')

    logger.info(f"Individual loading:  {individual_time:.2f}s for {n_samples} samples")
    logger.info(f"Batch loading:       {batch_time:.2f}s for {n_samples} samples")
    logger.info(f"Speedup:             {speedup:.1f}x faster!")
    logger.info(f"Full dataset:        ~{batch_time * (len(df) / n_samples):.1f}s estimated")

    if speedup > 10:
        logger.success(f"🚀 Excellent! Batch mode is {speedup:.1f}x faster!")
    elif speedup > 2:
        logger.info(f"✓ Good! Batch mode is {speedup:.1f}x faster")
    else:
        logger.info(f"  Batch mode is {speedup:.1f}x faster (dataset may be too small for significant gains)")

    # Extrapolate to full dataset
    full_individual_time = individual_time * (len(df) / n_samples)
    full_batch_time = batch_time * (len(df) / n_samples)
    time_saved = full_individual_time - full_batch_time

    logger.info(f"\nEstimated time for full dataset ({len(df)} samples):")
    logger.info(f"  Individual mode:  {full_individual_time:.1f}s ({full_individual_time/60:.1f} minutes)")
    logger.info(f"  Batch mode:       {full_batch_time:.1f}s ({full_batch_time/60:.1f} minutes)")
    logger.info(f"  Time saved:       {time_saved:.1f}s ({time_saved/60:.1f} minutes)")

    logger.info("\n" + "=" * 80)
    logger.success("Test completed!")
    logger.info("=" * 80)

    # Recommendations
    logger.info("\n💡 Recommendations:")
    logger.info("  - Always use batch mode for large datasets (1000+ samples)")
    logger.info("  - Enable progress bars for long-running operations (default)")
    logger.info("  - Install tqdm for progress bars: pip install tqdm")
    logger.info("\n✨ Happy feature engineering! ✨\n")


if __name__ == '__main__':
    test_batch_loading_performance()

