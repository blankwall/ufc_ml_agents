#!/usr/bin/env python3
"""
Compare Model Evaluation Results Against Baseline

This script compares a new evaluation JSON against a baseline JSON and provides
a verdict (ACCEPT/REJECT/REVIEW) based on predefined rules.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def safe_get(data: Dict, *keys, default=None):
    """Safely get nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def compare_accuracy(baseline: Dict, current: Dict) -> Tuple[str, str, float]:
    """
    Compare accuracy metrics.
    
    Returns: (verdict, reason, delta)
    - FAIL if accuracy decreases
    - PASS if accuracy increases by ≥ +1.0 pp
    - REVIEW if change is between –0.5 pp and +1.0 pp
    """
    baseline_acc = safe_get(baseline, "overall", "accuracy", default=0.0)
    current_acc = safe_get(current, "overall", "accuracy", default=0.0)
    
    if baseline_acc is None or current_acc is None:
        return "REVIEW", "Missing accuracy data", 0.0
    
    delta = current_acc - baseline_acc
    delta_pp = delta * 100  # Convert to percentage points
    
    if delta < 0:
        return "FAIL", f"Accuracy decreased by {abs(delta_pp):.2f} pp", delta_pp
    elif delta_pp >= 1.0:
        return "PASS", f"Accuracy increased by {delta_pp:.2f} pp", delta_pp
    elif delta_pp >= -0.5:
        return "REVIEW", f"Accuracy changed by {delta_pp:.2f} pp (marginal)", delta_pp
    else:
        return "FAIL", f"Accuracy decreased by {abs(delta_pp):.2f} pp", delta_pp


def compare_auc(baseline: Dict, current: Dict) -> Tuple[str, str, float]:
    """
    Compare AUC metrics.
    
    Returns: (verdict, reason, delta)
    - FAIL if AUC drops by > 0.01
    - PASS if AUC improves or stays within ±0.005
    - REVIEW if –0.01 ≤ Δ AUC ≤ –0.005
    """
    baseline_auc = safe_get(baseline, "overall", "auc", default=0.0)
    current_auc = safe_get(current, "overall", "auc", default=0.0)
    
    if baseline_auc is None or current_auc is None:
        return "REVIEW", "Missing AUC data", 0.0
    
    delta = current_auc - baseline_auc
    
    if delta < -0.01:
        return "FAIL", f"AUC dropped by {abs(delta):.4f}", delta
    elif delta >= -0.01 and delta < -0.005:
        return "REVIEW", f"AUC dropped by {abs(delta):.4f} (marginal)", delta
    elif delta >= -0.005 and delta <= 0.005:
        return "PASS", f"AUC changed by {delta:.4f} (within tolerance)", delta
    else:
        # delta > 0.005 (improved)
        return "PASS", f"AUC improved by {delta:.4f}", delta


def compare_brier(baseline: Dict, current: Dict) -> Tuple[str, str, float]:
    """
    Compare Brier score metrics.
    
    Returns: (verdict, reason, delta)
    - FAIL if Brier increases by > 0.005
    - PASS if Brier improves or stays within ±0.002
    - REVIEW otherwise
    """
    baseline_brier = safe_get(baseline, "overall", "brier", default=1.0)
    current_brier = safe_get(current, "overall", "brier", default=1.0)
    
    if baseline_brier is None or current_brier is None:
        return "REVIEW", "Missing Brier score data", 0.0
    
    delta = current_brier - baseline_brier  # Positive = worse (higher Brier is worse)
    
    if delta > 0.005:
        return "FAIL", f"Brier increased by {delta:.4f} (worse)", delta
    elif delta >= -0.002 and delta <= 0.002:
        return "PASS", f"Brier changed by {delta:.4f} (within tolerance)", delta
    elif delta < 0:
        return "PASS", f"Brier improved by {abs(delta):.4f}", delta
    else:
        return "REVIEW", f"Brier increased by {delta:.4f} (marginal)", delta


def compare_buckets(baseline: Dict, current: Dict) -> Tuple[str, list]:
    """
    Compare favorites/underdogs buckets.
    
    Returns: (worst_verdict, warnings)
    - REVIEW if either bucket drops by > 3 pp
    - FAIL only if: Overall accuracy ↑ AND one bucket ↓ > 5 pp
    """
    warnings = []
    worst_verdict = "PASS"
    
    baseline_fav_acc = safe_get(baseline, "by_bucket", "favorites", "accuracy", default=0.0)
    current_fav_acc = safe_get(current, "by_bucket", "favorites", "accuracy", default=0.0)
    baseline_und_acc = safe_get(baseline, "by_bucket", "underdogs", "accuracy", default=0.0)
    current_und_acc = safe_get(current, "by_bucket", "underdogs", "accuracy", default=0.0)
    
    baseline_overall_acc = safe_get(baseline, "overall", "accuracy", default=0.0)
    current_overall_acc = safe_get(current, "overall", "accuracy", default=0.0)
    
    if baseline_fav_acc is not None and current_fav_acc is not None:
        fav_delta_pp = (current_fav_acc - baseline_fav_acc) * 100
        if fav_delta_pp < -3:
            warnings.append(f"Favorites accuracy dropped by {abs(fav_delta_pp):.2f} pp")
            worst_verdict = "REVIEW"
            if fav_delta_pp < -5 and current_overall_acc > baseline_overall_acc:
                worst_verdict = "FAIL"
                warnings[-1] += " (FAIL: overall accuracy increased but favorites dropped >5pp)"
    
    if baseline_und_acc is not None and current_und_acc is not None:
        und_delta_pp = (current_und_acc - baseline_und_acc) * 100
        if und_delta_pp < -3:
            warnings.append(f"Underdogs accuracy dropped by {abs(und_delta_pp):.2f} pp")
            if worst_verdict == "PASS":
                worst_verdict = "REVIEW"
            if und_delta_pp < -5 and current_overall_acc > baseline_overall_acc:
                worst_verdict = "FAIL"
                warnings[-1] += " (FAIL: overall accuracy increased but underdogs dropped >5pp)"
    
    return worst_verdict, warnings


def compare_confidence(baseline: Dict, current: Dict) -> Tuple[str, list]:
    """
    Compare confidence buckets (top 10% and top 25%).
    
    Returns: (worst_verdict, warnings)
    - Top 10%: REVIEW if accuracy drops by > 5 pp, FAIL only if drop > 8 pp
    - Top 25%: REVIEW if accuracy drops by > 3 pp, FAIL only if drop > 5 pp
    """
    warnings = []
    worst_verdict = "PASS"
    
    # Top 10%
    baseline_top10 = safe_get(baseline, "by_confidence", "top_10_pct", default={})
    current_top10 = safe_get(current, "by_confidence", "top_10_pct", default={})
    
    baseline_top10_acc = baseline_top10.get("accuracy")
    current_top10_acc = current_top10.get("accuracy")
    
    if baseline_top10_acc is not None and current_top10_acc is not None:
        top10_delta_pp = (current_top10_acc - baseline_top10_acc) * 100
        if top10_delta_pp < -5:
            warnings.append(f"Top 10% accuracy dropped by {abs(top10_delta_pp):.2f} pp")
            worst_verdict = "REVIEW"
            if top10_delta_pp < -8:
                worst_verdict = "FAIL"
                warnings[-1] += " (FAIL: drop > 8pp)"
    
    # Top 25%
    baseline_top25 = safe_get(baseline, "by_confidence", "top_25_pct", default={})
    current_top25 = safe_get(current, "by_confidence", "top_25_pct", default={})
    
    baseline_top25_acc = baseline_top25.get("accuracy")
    current_top25_acc = current_top25.get("accuracy")
    
    if baseline_top25_acc is not None and current_top25_acc is not None:
        top25_delta_pp = (current_top25_acc - baseline_top25_acc) * 100
        if top25_delta_pp < -3:
            warnings.append(f"Top 25% accuracy dropped by {abs(top25_delta_pp):.2f} pp")
            if worst_verdict == "PASS":
                worst_verdict = "REVIEW"
            if top25_delta_pp < -5:
                worst_verdict = "FAIL"
                warnings[-1] += " (FAIL: drop > 5pp)"
    
    return worst_verdict, warnings


def check_data_consistency(baseline: Dict, current: Dict) -> list:
    """
    Check if data sizes changed materially.
    
    Returns: list of warnings (never FAIL)
    """
    warnings = []
    
    baseline_n = safe_get(baseline, "overall", "n_total", default=0)
    current_n = safe_get(current, "overall", "n_total", default=0)
    
    if baseline_n > 0 and current_n > 0:
        n_change_pct = abs(current_n - baseline_n) / baseline_n * 100
        if n_change_pct > 5:  # More than 5% change
            warnings.append(f"Evaluation rows changed: {baseline_n} → {current_n} ({n_change_pct:.1f}% change)")
    
    # Check bucket sizes
    baseline_fav_n = safe_get(baseline, "by_bucket", "favorites", "n", default=0)
    current_fav_n = safe_get(current, "by_bucket", "favorites", "n", default=0)
    baseline_und_n = safe_get(baseline, "by_bucket", "underdogs", "n", default=0)
    current_und_n = safe_get(current, "by_bucket", "underdogs", "n", default=0)
    
    if baseline_fav_n > 0 and current_fav_n > 0:
        fav_change_pct = abs(current_fav_n - baseline_fav_n) / baseline_fav_n * 100
        if fav_change_pct > 10:
            warnings.append(f"Favorites bucket size changed: {baseline_fav_n} → {current_fav_n} ({fav_change_pct:.1f}% change)")
    
    if baseline_und_n > 0 and current_und_n > 0:
        und_change_pct = abs(current_und_n - baseline_und_n) / baseline_und_n * 100
        if und_change_pct > 10:
            warnings.append(f"Underdogs bucket size changed: {baseline_und_n} → {current_und_n} ({und_change_pct:.1f}% change)")
    
    # Check confidence bucket sizes
    baseline_top10_n = safe_get(baseline, "by_confidence", "top_10_pct", "n", default=0)
    current_top10_n = safe_get(current, "by_confidence", "top_10_pct", "n", default=0)
    baseline_top25_n = safe_get(baseline, "by_confidence", "top_25_pct", "n", default=0)
    current_top25_n = safe_get(current, "by_confidence", "top_25_pct", "n", default=0)
    
    if baseline_top10_n > 0 and current_top10_n > 0:
        top10_change_pct = abs(current_top10_n - baseline_top10_n) / baseline_top10_n * 100
        if top10_change_pct > 20:
            warnings.append(f"Top 10% bucket size changed: {baseline_top10_n} → {current_top10_n} ({top10_change_pct:.1f}% change)")
    
    if baseline_top25_n > 0 and current_top25_n > 0:
        top25_change_pct = abs(current_top25_n - baseline_top25_n) / baseline_top25_n * 100
        if top25_change_pct > 20:
            warnings.append(f"Top 25% bucket size changed: {baseline_top25_n} → {current_top25_n} ({top25_change_pct:.1f}% change)")
    
    return warnings


def determine_verdict(results: Dict) -> Tuple[str, str]:
    """
    Determine final verdict based on all comparison results.
    
    ACCEPT if:
    - Accuracy PASS
    - AUC PASS
    - Brier PASS
    - No FAIL in buckets or confidence
    
    REJECT if:
    - Any hard FAIL
    
    REVIEW if:
    - Accuracy marginal
    - OR multiple soft warnings
    """
    accuracy_verdict = results["accuracy"]["verdict"]
    auc_verdict = results["auc"]["verdict"]
    brier_verdict = results["brier"]["verdict"]
    buckets_verdict = results["buckets"]["verdict"]
    confidence_verdict = results["confidence"]["verdict"]
    
    # Check for any hard FAIL
    if accuracy_verdict == "FAIL" or auc_verdict == "FAIL" or brier_verdict == "FAIL":
        return "REJECT", "Hard FAIL detected in core metrics"
    
    if buckets_verdict == "FAIL" or confidence_verdict == "FAIL":
        return "REJECT", "Hard FAIL detected in bucket or confidence analysis"
    
    # Check if all core metrics PASS
    if accuracy_verdict == "PASS" and auc_verdict == "PASS" and brier_verdict == "PASS":
        if buckets_verdict != "FAIL" and confidence_verdict != "FAIL":
            return "ACCEPT", "All core metrics PASS, no bucket/confidence FAILs"
    
    # Otherwise REVIEW
    reasons = []
    if accuracy_verdict == "REVIEW":
        reasons.append("Accuracy marginal")
    if auc_verdict == "REVIEW":
        reasons.append("AUC marginal")
    if brier_verdict == "REVIEW":
        reasons.append("Brier marginal")
    if buckets_verdict == "REVIEW":
        reasons.append("Bucket analysis issues")
    if confidence_verdict == "REVIEW":
        reasons.append("Confidence bucket issues")
    
    all_warnings = results["buckets"]["warnings"] + results["confidence"]["warnings"] + results["data_consistency"]
    if len(all_warnings) > 2:
        reasons.append(f"Multiple warnings ({len(all_warnings)} total)")
    
    return "REVIEW", "; ".join(reasons) if reasons else "Marginal changes detected"


def main():
    parser = argparse.ArgumentParser(
        description="Compare model evaluation results against baseline"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="models/baseline.json",
        help="Path to baseline JSON file (default: models/baseline.json)",
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current evaluation JSON file to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save comparison report JSON",
    )
    
    args = parser.parse_args()
    
    # Load files
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    
    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        return
    
    if not current_path.exists():
        logger.error(f"Current evaluation file not found: {current_path}")
        return
    
    logger.info(f"Loading baseline from {baseline_path}")
    baseline = load_json(baseline_path)
    
    logger.info(f"Loading current evaluation from {current_path}")
    current = load_json(current_path)
    
    # Run comparisons
    logger.info("Running comparisons...")
    
    acc_verdict, acc_reason, acc_delta = compare_accuracy(baseline, current)
    auc_verdict, auc_reason, auc_delta = compare_auc(baseline, current)
    brier_verdict, brier_reason, brier_delta = compare_brier(baseline, current)
    buckets_verdict, buckets_warnings = compare_buckets(baseline, current)
    confidence_verdict, confidence_warnings = compare_confidence(baseline, current)
    data_warnings = check_data_consistency(baseline, current)
    
    results = {
        "accuracy": {
            "verdict": acc_verdict,
            "reason": acc_reason,
            "delta": acc_delta,
            "baseline": safe_get(baseline, "overall", "accuracy"),
            "current": safe_get(current, "overall", "accuracy"),
        },
        "auc": {
            "verdict": auc_verdict,
            "reason": auc_reason,
            "delta": auc_delta,
            "baseline": safe_get(baseline, "overall", "auc"),
            "current": safe_get(current, "overall", "auc"),
        },
        "brier": {
            "verdict": brier_verdict,
            "reason": brier_reason,
            "delta": brier_delta,
            "baseline": safe_get(baseline, "overall", "brier"),
            "current": safe_get(current, "overall", "brier"),
        },
        "buckets": {
            "verdict": buckets_verdict,
            "warnings": buckets_warnings,
        },
        "confidence": {
            "verdict": confidence_verdict,
            "warnings": confidence_warnings,
        },
        "data_consistency": data_warnings,
    }
    
    # Determine final verdict
    final_verdict, final_reason = determine_verdict(results)
    results["final_verdict"] = final_verdict
    results["final_reason"] = final_reason
    
    # Print results
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nBaseline: {baseline_path}")
    print(f"Current:  {current_path}")
    print(f"\n{'=' * 80}")
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"Reason: {final_reason}")
    print(f"{'=' * 80}\n")
    
    print("Core Metrics:")
    print(f"  Accuracy: {acc_verdict:6s} - {acc_reason}")
    print(f"  AUC:      {auc_verdict:6s} - {auc_reason}")
    print(f"  Brier:    {brier_verdict:6s} - {brier_reason}")
    
    # Bucket Analysis with percentage changes
    print(f"\nBucket Analysis: {buckets_verdict}")
    baseline_fav_acc = safe_get(baseline, "by_bucket", "favorites", "accuracy")
    current_fav_acc = safe_get(current, "by_bucket", "favorites", "accuracy")
    baseline_und_acc = safe_get(baseline, "by_bucket", "underdogs", "accuracy")
    current_und_acc = safe_get(current, "by_bucket", "underdogs", "accuracy")
    
    if baseline_fav_acc is not None and current_fav_acc is not None:
        fav_delta_pp = (current_fav_acc - baseline_fav_acc) * 100
        fav_sign = "+" if fav_delta_pp >= 0 else ""
        print(f"  Favorites: {fav_sign}{fav_delta_pp:.1f}%  {buckets_verdict}")
    
    if baseline_und_acc is not None and current_und_acc is not None:
        und_delta_pp = (current_und_acc - baseline_und_acc) * 100
        und_sign = "+" if und_delta_pp >= 0 else ""
        print(f"  Underdogs: {und_sign}{und_delta_pp:.1f}%  {buckets_verdict}")
    
    if buckets_warnings:
        for warning in buckets_warnings:
            print(f"  ⚠️  {warning}")
    
    # Confidence Analysis with percentage changes
    print(f"\nConfidence Analysis: {confidence_verdict}")
    baseline_top10 = safe_get(baseline, "by_confidence", "top_10_pct", default={})
    current_top10 = safe_get(current, "by_confidence", "top_10_pct", default={})
    baseline_top25 = safe_get(baseline, "by_confidence", "top_25_pct", default={})
    current_top25 = safe_get(current, "by_confidence", "top_25_pct", default={})
    
    baseline_top10_acc = baseline_top10.get("accuracy")
    current_top10_acc = current_top10.get("accuracy")
    baseline_top25_acc = baseline_top25.get("accuracy")
    current_top25_acc = current_top25.get("accuracy")
    
    if baseline_top10_acc is not None and current_top10_acc is not None:
        top10_delta_pp = (current_top10_acc - baseline_top10_acc) * 100
        top10_sign = "+" if top10_delta_pp >= 0 else ""
        print(f"  Top 10%: {top10_sign}{top10_delta_pp:.1f}%  {confidence_verdict}")
    
    if baseline_top25_acc is not None and current_top25_acc is not None:
        top25_delta_pp = (current_top25_acc - baseline_top25_acc) * 100
        top25_sign = "+" if top25_delta_pp >= 0 else ""
        print(f"  Top 25%: {top25_sign}{top25_delta_pp:.1f}%  {confidence_verdict}")
    
    if confidence_warnings:
        for warning in confidence_warnings:
            print(f"  ⚠️  {warning}")
    
    if data_warnings:
        print(f"\nData Consistency Warnings:")
        for warning in data_warnings:
            print(f"  ⚠️  {warning}")
    
    print("\n" + "=" * 80)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Saved comparison report to {output_path}")
    
    # Exit with appropriate code
    if final_verdict == "REJECT":
        exit(1)
    elif final_verdict == "REVIEW":
        exit(2)
    else:
        exit(0)


if __name__ == "__main__":
    main()

