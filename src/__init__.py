"""
Multi-omics Biomarker Discovery for Drug Response Prediction

A comprehensive research pipeline for integrating genomics, transcriptomics, 
and drug sensitivity data from GDSC to discover biomarkers for drug response prediction.

This package provides:
- Multi-omics data acquisition and preprocessing
- Advanced data integration strategies
- Machine learning models for drug response prediction
- Biomarker discovery and validation
- Pathway enrichment analysis
- Comprehensive visualization and reporting

Author: Research Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@institution.edu"
__license__ = "MIT"

# Core modules
from . import data_acquisition
from . import preprocessing
from . import integration
from . import modeling
from . import pathway_analysis
from . import biomarker_discovery
from . import visualization
from . import utils

# Main classes for easy import
from .data_acquisition import GDSCDataAcquisition, DataAcquisitionManager
from .preprocessing import MultiOmicsPreprocessor
from .integration import MultiOmicsIntegrator
from .modeling import DrugResponsePredictor
from .pathway_analysis import PathwayAnalyzer
from .biomarker_discovery import BiomarkerDiscovery
from .utils import setup_logging, load_config

__all__ = [
    # Modules
    "data_acquisition",
    "preprocessing", 
    "integration",
    "modeling",
    "pathway_analysis",
    "biomarker_discovery",
    "visualization",
    "utils",
    
    # Main classes
    "GDSCDataAcquisition",
    "DataAcquisitionManager",
    "MultiOmicsPreprocessor",
    "MultiOmicsIntegrator", 
    "DrugResponsePredictor",
    "PathwayAnalyzer",
    "BiomarkerDiscovery",
    
    # Utilities
    "setup_logging",
    "load_config",
]

# Package metadata
__package_info__ = {
    "name": "multiomics-biomarker-discovery",
    "version": __version__,
    "description": "Multi-omics Biomarker Discovery for Drug Response Prediction",
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/your-username/multiomics-biomarker-discovery",
    "keywords": [
        "bioinformatics",
        "multi-omics", 
        "biomarkers",
        "drug response",
        "machine learning",
        "pathway analysis",
        "genomics",
        "transcriptomics"
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
}