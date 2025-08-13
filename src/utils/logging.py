"""
Logging utilities for Multi-omics Biomarker Discovery

This module provides logging setup and configuration functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..exceptions import ConfigurationError


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs to console only.
        format_string: Custom format string for log messages
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        ConfigurationError: If logging configuration fails
    """
    try:
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create formatter
        formatter = logging.Formatter(format_string)
        
        # Get root logger
        logger = logging.getLogger("multiomics")
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
        
    except Exception as e:
        raise ConfigurationError(f"Failed to setup logging: {str(e)}")


__all__ = ["setup_logging"]