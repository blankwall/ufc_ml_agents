"""
Feature Engineering Utilities
Shared helper functions for feature extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime


def safe_mean(values: List[float]) -> float:
    """
    Safely compute mean of a list, returning 0.0 for empty lists.
    
    Args:
        values: List of numeric values
        
    Returns:
        Mean value or 0.0 if empty
    """
    if not values or len(values) == 0:
        return 0.0
    try:
        return float(np.mean(values))
    except (TypeError, ValueError):
        return 0.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero or invalid.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if division fails
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return float(numerator / denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def parse_landed(value: Optional[str]) -> float:
    """
    Parse strings like '42 of 60' -> 42.0.
    
    Args:
        value: String in format "X of Y"
        
    Returns:
        Parsed numeric value or 0.0 on failure
    """
    if not value:
        return 0.0
    try:
        parts = str(value).split(" of ")
        return float(parts[0].strip())
    except (ValueError, IndexError, AttributeError):
        return 0.0


def parse_int(value: Optional[str]) -> int:
    """
    Safely parse an integer field, defaulting to 0.
    
    Args:
        value: Value to parse
        
    Returns:
        Parsed integer or 0
    """
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except (ValueError, AttributeError):
        return 0


def parse_control_time_seconds(value: Optional[str]) -> float:
    """
    Parse control time like '6:02' -> 362 seconds.
    Accepts either 'MM:SS' or raw numeric seconds string.
    
    Args:
        value: Control time string
        
    Returns:
        Total seconds or 0.0 on failure
    """
    if not value:
        return 0.0
    try:
        text = str(value).strip()
        if ":" not in text:
            return float(text)
        mins_str, secs_str = text.split(":", 1)
        mins = int(mins_str.strip())
        secs = int(secs_str.strip())
        return float(mins * 60 + secs)
    except (ValueError, AttributeError):
        return 0.0


def is_finish(method: Optional[str]) -> bool:
    """
    Check if a fight method indicates a finish (KO/TKO/Submission).
    
    Args:
        method: Fight method string
        
    Returns:
        True if method is a finish
    """
    if not method:
        return False
    method_str = str(method).upper()
    return any(term in method_str for term in ["KO", "TKO", "SUB", "SUBMISSION"])


def is_ko(method: Optional[str]) -> bool:
    """
    Check if a fight method indicates a KO/TKO.
    
    Args:
        method: Fight method string
        
    Returns:
        True if method is a KO/TKO
    """
    if not method:
        return False
    method_str = str(method).upper()
    return "KO" in method_str or "TKO" in method_str


def is_submission(method: Optional[str]) -> bool:
    """
    Check if a fight method indicates a submission.
    
    Args:
        method: Fight method string
        
    Returns:
        True if method is a submission
    """
    if not method:
        return False
    method_str = str(method).upper()
    return "SUB" in method_str or "SUBMISSION" in method_str


def is_decision(method: Optional[str]) -> bool:
    """
    Check if a fight method indicates a decision.
    
    Args:
        method: Fight method string
        
    Returns:
        True if method is a decision
    """
    if not method:
        return False
    method_str = str(method).upper()
    return "DEC" in method_str or "DECISION" in method_str


def calculate_rolling_rate(
    df: pd.DataFrame,
    window: int,
    condition_col: str,
    condition_value: str,
    total_col: str = "result"
) -> float:
    """
    Calculate rolling rate for a condition over a window.
    
    Args:
        df: DataFrame with fight history (sorted most recent first)
        window: Number of fights to consider
        condition_col: Column to check condition on
        condition_value: Value to match
        total_col: Column to use for total count (default: "result")
        
    Returns:
        Rate as float between 0 and 1
    """
    if len(df) == 0:
        return 0.0
    
    recent = df.head(window)
    if len(recent) == 0:
        return 0.0
    
    matches = (recent[condition_col] == condition_value).sum()
    return safe_divide(matches, len(recent))


def calculate_time_decayed_metric(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    lambda_decay: float = 0.3,
    reference_date: Optional[datetime] = None
) -> float:
    """
    Calculate time-decayed metric where recent values are weighted more heavily.
    
    Uses exponential decay: weight = exp(-lambda * years_ago)
    
    Args:
        df: DataFrame with fight history
        date_col: Column name for dates
        value_col: Column name for values to weight
        lambda_decay: Decay rate (higher = faster decay)
        reference_date: Reference date for calculating years_ago (default: now)
        
    Returns:
        Weighted average value
    """
    if len(df) == 0 or date_col not in df.columns or value_col not in df.columns:
        return 0.0
    
    if reference_date is None:
        reference_date = datetime.now()
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for _, row in df.iterrows():
        try:
            fight_date = row[date_col]
            if pd.isna(fight_date):
                continue
            
            years_ago = (reference_date - fight_date).days / 365.25
            weight = np.exp(-lambda_decay * years_ago)
            
            value = row[value_col]
            if pd.notna(value):
                total_weight += weight
                weighted_sum += weight * float(value)
        except (TypeError, ValueError, KeyError):
            continue
    
    return safe_divide(weighted_sum, total_weight)


def ensure_numeric(value: any, default: float = 0.0) -> float:
    """
    Ensure a value is numeric, converting if necessary.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Numeric value
    """
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

