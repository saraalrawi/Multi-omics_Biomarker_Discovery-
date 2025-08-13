"""
Base classes for data acquisition components

This module provides abstract base classes for data acquisition functionality.
"""

from abc import ABC, abstractmethod


class DataAcquisitionBase(ABC):
    """Abstract base class for data acquisition components."""
    
    @abstractmethod
    def download(self, **kwargs) -> bool:
        """
        Download data from source.
        
        Args:
            **kwargs: Additional arguments for downloading
            
        Returns:
            bool: True if download successful
        """
        pass
    
    @abstractmethod
    def validate(self, **kwargs) -> bool:
        """
        Validate downloaded data.
        
        Args:
            **kwargs: Additional arguments for validation
            
        Returns:
            bool: True if validation successful
        """
        pass


__all__ = ["DataAcquisitionBase"]