"""
Filesystem utilities for Multi-omics Biomarker Discovery

This module provides filesystem operations including directory management.
"""

from pathlib import Path
from typing import Union

from ..exceptions import FileOperationError


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path: Path object for the directory
        
    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise FileOperationError(f"Failed to create directory {path}: {str(e)}")


__all__ = ["ensure_directory"]