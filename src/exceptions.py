"""
Custom exception classes for Multi-omics Biomarker Discovery

This module defines project-specific exceptions to provide better error handling
and debugging capabilities throughout the application.
"""

from typing import Optional, Any


class MultiomicsError(Exception):
    """Base exception class for all multi-omics related errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class ConfigurationError(MultiomicsError):
    """Raised when there are configuration-related issues."""
    pass


class DataValidationError(MultiomicsError):
    """Raised when data validation fails."""
    pass


class DataAcquisitionError(MultiomicsError):
    """Raised when data acquisition fails."""
    pass


class PreprocessingError(MultiomicsError):
    """Raised when preprocessing operations fail."""
    pass


class IntegrationError(MultiomicsError):
    """Raised when multi-omics integration fails."""
    pass


class ModelingError(MultiomicsError):
    """Raised when modeling operations fail."""
    pass


class PathwayAnalysisError(MultiomicsError):
    """Raised when pathway analysis fails."""
    pass


class BiomarkerDiscoveryError(MultiomicsError):
    """Raised when biomarker discovery fails."""
    pass


class VisualizationError(MultiomicsError):
    """Raised when visualization operations fail."""
    pass


class APIError(MultiomicsError):
    """Raised when external API calls fail."""
    pass


class FileOperationError(MultiomicsError):
    """Raised when file operations fail."""
    pass


# Export all exception classes
__all__ = [
    "MultiomicsError",
    "ConfigurationError", 
    "DataValidationError",
    "DataAcquisitionError",
    "PreprocessingError",
    "IntegrationError",
    "ModelingError",
    "PathwayAnalysisError",
    "BiomarkerDiscoveryError",
    "VisualizationError",
    "APIError",
    "FileOperationError",
]