"""
Data acquisition management functionality

This module provides the DataAcquisitionManager class for coordinating
data acquisition from multiple sources.
"""

import logging
from typing import Dict, Any, Optional

from .gdsc import GDSCDataAcquisition
from ..utils import load_config
from ..exceptions import DataAcquisitionError

logger = logging.getLogger(__name__)


class DataAcquisitionManager:
    """
    High-level manager for coordinating data acquisition from multiple sources.
    
    This class orchestrates the acquisition of multi-omics data from various sources
    and ensures data consistency and completeness.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize data acquisition manager.
        
        Args:
            config: Configuration object
            
        Raises:
            DataAcquisitionError: If initialization fails
        """
        try:
            self.config = config or load_config()
            self.gdsc_client = GDSCDataAcquisition(config)
            
            logger.info("Initialized DataAcquisitionManager")
        except Exception as e:
            raise DataAcquisitionError(f"Failed to initialize DataAcquisitionManager: {str(e)}")
    
    def acquire_all_data(self, force_download: bool = False) -> Dict[str, bool]:
        """
        Acquire all configured data sources.
        
        Args:
            force_download: Whether to re-download existing data
            
        Returns:
            Dict[str, bool]: Status of each data source acquisition
            
        Raises:
            DataAcquisitionError: If acquisition process fails critically
        """
        try:
            results = {}
            
            # Acquire GDSC data
            logger.info("Starting GDSC data acquisition...")
            results['gdsc'] = self.gdsc_client.download(force_download=force_download)
            
            # Validate acquired data
            if results['gdsc']:
                results['gdsc_validation'] = self.gdsc_client.validate()
            else:
                results['gdsc_validation'] = False
            
            # Log summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"Data acquisition completed: {successful}/{total} successful")
            
            return results
        except Exception as e:
            raise DataAcquisitionError(f"Data acquisition process failed: {str(e)}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data.
        
        Returns:
            Dict[str, Any]: Summary of data availability and statistics
        """
        try:
            summary = {
                'gdsc': {
                    'available_drugs': self.gdsc_client.get_available_drugs(),
                    'available_cancer_types': self.gdsc_client.get_available_cancer_types(),
                    'data_path': str(self.gdsc_client.output_dir)
                }
            }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to generate data summary: {str(e)}")
            return {}
    
    def validate_all_data(self) -> Dict[str, bool]:
        """
        Validate all acquired data sources.
        
        Returns:
            Dict[str, bool]: Validation status for each data source
        """
        try:
            results = {}
            
            # Validate GDSC data
            logger.info("Validating GDSC data...")
            results['gdsc'] = self.gdsc_client.validate()
            
            # Log validation summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"Data validation completed: {successful}/{total} passed")
            
            return results
        except Exception as e:
            logger.error(f"Data validation process failed: {str(e)}")
            return {}


__all__ = ["DataAcquisitionManager"]