"""
Data Acquisition Module for Multi-omics Biomarker Discovery

This module provides functionality for acquiring and managing multi-omics data from various sources,
with a focus on GDSC (Genomics of Drug Sensitivity in Cancer) data.

Key components:
- GDSCDataAcquisition: Main class for downloading and processing GDSC data
- DataAcquisitionManager: High-level manager for coordinating data acquisition tasks
- DataAcquisitionBase: Abstract base class for data acquisition components
"""

from .base import DataAcquisitionBase
from .gdsc import GDSCDataAcquisition
from .manager import DataAcquisitionManager

# Export main classes
__all__ = [
    "DataAcquisitionBase",
    "GDSCDataAcquisition",
    "DataAcquisitionManager"
]