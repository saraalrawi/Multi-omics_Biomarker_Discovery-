"""
Helper utilities for Multi-omics Biomarker Discovery

This module provides general helper functions and utilities.
"""

from datetime import datetime


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime(format_string)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        float: Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


__all__ = [
    "get_timestamp",
    "safe_divide"
]