"""
Performance Tools - Utilities for benchmarking and profiling feature extraction

This module provides tools to:
- Profile feature extraction bottlenecks
- Benchmark optimizations
- Compare performance between versions
"""

import time
import cProfile
import pstats
from io import StringIO
from contextlib import contextmanager
from typing import Callable, Any, Dict, List
import numpy as np
from loguru import logger


@contextmanager
def timer(name: str, log_level: str = "info"):
    """
    Context manager to time operations and log the result.

    Args:
        name: Name of the operation being timed
        log_level: Log level ('debug', 'info', 'warning', 'error')

    Example:
        with timer("Dataset Creation"):
            create_training_dataset(session)
    """
    start = time.time()
    yield
    elapsed = time.time() - start

    log_func = getattr(logger, log_level.lower(), logger.info)
    log_func(f"[PERF] {name}: {elapsed:.2f}s")


def benchmark(func: Callable, *args, iterations: int = 10, **kwargs) -> float:
    """
    Benchmark a function multiple times and return average time.

    Args:
        func: Function to benchmark
        *args: Positional arguments to pass to function
        iterations: Number of times to run the function (default: 10)
        **kwargs: Keyword arguments to pass to function

    Returns:
        Average execution time in seconds

    Example:
        avg_time = benchmark(create_training_dataset, session, iterations=5)
        logger.info(f"Average time: {avg_time:.2f}s")
    """
    times = []
    for i in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.debug(f"  Run {i+1}/{iterations}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = np.std(times)

    logger.info(f"[BENCHMARK] {func.__name__}:")
    logger.info(f"  Average: {avg:.2f}s")
    logger.info(f"  Min: {min_time:.2f}s")
    logger.info(f"  Max: {max_time:.2f}s")
    logger.info(f"  Std Dev: {std_dev:.2f}s")
    logger.info(f"  Iterations: {iterations}")

    return avg


def profile(func: Callable, *args, num_lines: int = 30, **kwargs):
    """
    Profile a function and print the top slowest operations.

    Args:
        func: Function to profile
        *args: Positional arguments to pass to function
        num_lines: Number of top lines to show (default: 30)
        **kwargs: Keyword arguments to pass to function

    Example:
        profile(create_training_dataset, session)
    """
    pr = cProfile.Profile()
    pr.enable()

    # Run the function
    func(*args, **kwargs)

    pr.disable()

    # Print results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(num_lines)

    logger.info(f"[PROFILE] Top {num_lines} slowest operations for {func.__name__}:")
    logger.info("\n" + s.getvalue())


def compare_performance(
    baseline_func: Callable,
    optimized_func: Callable,
    *args,
    iterations: int = 5,
    **kwargs
) -> Dict[str, float]:
    """
    Compare performance between baseline and optimized functions.

    Args:
        baseline_func: Baseline function
        optimized_func: Optimized function
        *args: Positional arguments to pass to both functions
        iterations: Number of iterations for each function (default: 5)
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Dictionary with performance metrics

    Example:
        results = compare_performance(
            extract_features_baseline,
            extract_features_optimized,
            fighter_id
        )
        speedup = results['speedup']
        logger.info(f"Speedup: {speedup:.2f}x")
    """
    logger.info("=" * 80)
    logger.info(f"PERFORMANCE COMPARISON: {baseline_func.__name__} vs {optimized_func.__name__}")
    logger.info("=" * 80)

    # Benchmark baseline
    logger.info(f"\n📊 Benchmarking baseline: {baseline_func.__name__}")
    baseline_times = []
    baseline_results = []
    for i in range(iterations):
        start = time.time()
        result = baseline_func(*args, **kwargs)
        baseline_times.append(time.time() - start)
        baseline_results.append(result)

    baseline_avg = sum(baseline_times) / len(baseline_times)

    # Benchmark optimized
    logger.info(f"\n🚀 Benchmarking optimized: {optimized_func.__name__}")
    optimized_times = []
    optimized_results = []
    for i in range(iterations):
        start = time.time()
        result = optimized_func(*args, **kwargs)
        optimized_times.append(time.time() - start)
        optimized_results.append(result)

    optimized_avg = sum(optimized_times) / len(optimized_times)

    # Calculate metrics
    speedup = baseline_avg / optimized_avg if optimized_avg > 0 else float('inf')
    improvement_pct = ((baseline_avg - optimized_avg) / baseline_avg) * 100

    results = {
        'baseline_avg': baseline_avg,
        'optimized_avg': optimized_avg,
        'speedup': speedup,
        'improvement_pct': improvement_pct,
        'time_saved': baseline_avg - optimized_avg
    }

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Baseline average:  {baseline_avg:.4f}s")
    logger.info(f"Optimized average: {optimized_avg:.4f}s")
    logger.info(f"Time saved:       {results['time_saved']:.4f}s")
    logger.info(f"Speedup:         {speedup:.2f}x")
    logger.info(f"Improvement:      {improvement_pct:.2f}%")
    logger.info("=" * 80)

    # Verify correctness (if results are comparable)
    if len(baseline_results) > 0 and len(optimized_results) > 0:
        baseline_sample = baseline_results[0]
        optimized_sample = optimized_results[0]

        if isinstance(baseline_sample, dict) and isinstance(optimized_sample, dict):
            # Compare dictionaries
            baseline_arr = np.array(list(baseline_sample.values()))
            optimized_arr = np.array(list(optimized_sample.values()))

            if baseline_arr.shape == optimized_arr.shape:
                are_close = np.allclose(baseline_arr, optimized_arr, rtol=1e-5)
                if are_close:
                    logger.success("✅ Results are verified identical (within floating point tolerance)")
                else:
                    max_diff = np.max(np.abs(baseline_arr - optimized_arr))
                    logger.warning(f"⚠️  Results differ by up to {max_diff:.10f}")

    return results


def track_memory_usage(func: Callable, *args, **kwargs):
    """
    Track memory usage during function execution.

    Requires psutil: pip install psutil

    Args:
        func: Function to track
        *args: Positional arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Example:
        track_memory_usage(create_training_dataset, session)
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"[MEMORY] Before {func.__name__}: {mem_before:.2f} MB")

        # Run function
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        # Get final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        logger.info(f"[MEMORY] After {func.__name__}: {mem_after:.2f} MB")
        logger.info(f"[MEMORY] Peak memory used: {mem_used:.2f} MB")
        logger.info(f"[TIME] {func.__name__}: {elapsed:.2f}s")

        return result

    except ImportError:
        logger.warning("psutil not installed. Install with: pip install psutil")
        return func(*args, **kwargs)


def estimate_dataset_creation_time(
    num_fights: int,
    sample_time: float,
    sample_size: int
):
    """
    Estimate total dataset creation time based on sample.

    Args:
        num_fights: Total number of fights to process
        sample_time: Time to process sample fights
        sample_size: Number of fights in sample

    Example:
        sample_time = measure_sample_time(session, sample_size=100)
        estimate_dataset_creation_time(5000, sample_time, 100)
    """
    estimated_total_time = sample_time * (num_fights / sample_size)
    estimated_minutes = estimated_total_time / 60
    estimated_hours = estimated_total_time / 3600

    logger.info("=" * 80)
    logger.info("DATASET CREATION TIME ESTIMATE")
    logger.info("=" * 80)
    logger.info(f"Total fights: {num_fights}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Sample time: {sample_time:.2f}s")
    logger.info(f"\nEstimated time:")
    logger.info(f"  Total: {estimated_total_time:.2f}s")
    logger.info(f"  Minutes: {estimated_minutes:.2f}m")
    logger.info(f"  Hours: {estimated_hours:.2f}h")
    logger.info("=" * 80)

    return estimated_total_time


def measure_sample_time(
    create_func: Callable,
    session,
    sample_size: int = 100
) -> float:
    """
    Measure time to process a sample of fights.

    Args:
        create_func: Function to call (e.g., create_training_dataset)
        session: Database session
        sample_size: Number of fights to sample

    Returns:
        Time in seconds to process sample
    """
    # Get total fights
    from database.schema import Fight
    total_fights = session.query(Fight).filter(Fight.result != None).count()
    sample_size = min(sample_size, total_fights)

    logger.info(f"Measuring sample time with {sample_size}/{total_fights} fights")

    start = time.time()

    # Process sample (simplified - you'd need to modify create_func to support limit)
    fights = (
        session.query(Fight)
        .filter(Fight.result != None)
        .join(Fight.event)
        .order_by(Fight.id)
        .limit(sample_size)
        .all()
    )

    # ... process fights ...

    elapsed = time.time() - start

    logger.info(f"Sample time: {elapsed:.2f}s")

    return elapsed


# Example usage functions
def example_usage():
    """Examples of how to use performance tools"""

    # 1. Simple timing
    from database.db_manager import DatabaseManager

    db = DatabaseManager()
    session = db.get_session()

    with timer("Load Dataset"):
        from features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = pipeline.load_dataset()

    # 2. Benchmarking
    def dummy_function():
        import time
        time.sleep(0.1)

    avg_time = benchmark(dummy_function, iterations=5)
    print(f"\nAverage time: {avg_time:.2f}s")

    # 3. Performance comparison
    def baseline_func(x):
        # Simulate slow operation
        result = []
        for i in range(1000):
            result.append(i ** 2)
        return result

    def optimized_func(x):
        # Simulate fast operation
        return [i ** 2 for i in range(1000)]

    results = compare_performance(baseline_func, optimized_func, None)
    print(f"\nSpeedup: {results['speedup']:.2f}x")

    session.close()


if __name__ == '__main__':
    example_usage()

