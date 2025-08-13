"""
Configuration management for Multi-omics Biomarker Discovery

This module provides hierarchical configuration management with support for:
- YAML configuration files
- Environment variable overrides
- Validation and type checking
- Development/production environments
"""

from .settings import ProjectConfig, load_config, validate_config
from .data_sources import DataSourceConfig
from .model_config import ModelConfig
from .pathway_config import PathwayConfig

__all__ = [
    "ProjectConfig",
    "DataSourceConfig", 
    "ModelConfig",
    "PathwayConfig",
    "load_config",
    "validate_config",
]