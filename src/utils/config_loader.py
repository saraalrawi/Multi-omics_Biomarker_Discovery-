"""
Configuration loading utilities for Multi-omics Biomarker Discovery

This module provides configuration loading functionality as a wrapper
around the main configuration system.
"""

from pathlib import Path
from typing import Optional, Union

from ..exceptions import ConfigurationError


def load_config(config_path: Optional[Union[str, Path]] = None):
    """
    Load project configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ProjectConfig: Loaded configuration object
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    try:
        # Import here to avoid circular imports
        from ...config.settings import load_config as _load_config
        return _load_config(config_path)
    except ImportError:
        # Fallback import for direct usage
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from config.settings import load_config as _load_config
            return _load_config(config_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration module: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


__all__ = ["load_config"]