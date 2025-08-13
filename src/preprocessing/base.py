"""
Base classes for preprocessing components

This module provides abstract base classes for preprocessing functionality.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self


class PreprocessorBase(ABC):
    """Abstract base class for preprocessing components."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> Self:
        """
        Fit the preprocessor to data.
        
        Args:
            data: Input data to fit the preprocessor
            **kwargs: Additional fitting parameters
            
        Returns:
            Self: Fitted preprocessor instance
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Input data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            pd.DataFrame: Transformed data
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)


__all__ = ["PreprocessorBase"]