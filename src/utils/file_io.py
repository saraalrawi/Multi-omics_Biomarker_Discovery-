"""
File I/O utilities for Multi-omics Biomarker Discovery

This module provides file input/output operations including YAML handling
and file validation.
"""

from pathlib import Path
from typing import Dict, Any, Union
import yaml

from ..exceptions import FileOperationError, DataValidationError


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        
    Raises:
        FileOperationError: If file saving fails
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise FileOperationError(f"Failed to save YAML file {file_path}: {str(e)}")


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dict[str, Any]: Loaded data
        
    Raises:
        FileOperationError: If file loading fails
    """
    try:
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileOperationError(f"YAML file not found: {file_path}")
    except yaml.YAMLError as e:
        raise FileOperationError(f"Invalid YAML format in {file_path}: {str(e)}")
    except Exception as e:
        raise FileOperationError(f"Failed to load YAML file {file_path}: {str(e)}")


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and return a Path object.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Path: Validated path object
        
    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        DataValidationError: If path is invalid
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return path
    except FileNotFoundError:
        raise  # Re-raise FileNotFoundError as-is
    except Exception as e:
        raise DataValidationError(f"Invalid file path {file_path}: {str(e)}")


__all__ = [
    "save_yaml",
    "load_yaml", 
    "validate_file_path"
]