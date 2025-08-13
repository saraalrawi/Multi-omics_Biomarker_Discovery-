"""
Multi-omics preprocessing functionality

This module provides the MultiOmicsPreprocessor class for comprehensive
preprocessing of different omics data types.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
from scipy import stats
from scipy.stats import zscore, pearsonr
import warnings

from .base import PreprocessorBase
from ..utils import load_config, ensure_directory
from ..exceptions import PreprocessingError, DataValidationError

logger = logging.getLogger(__name__)


class MultiOmicsPreprocessor(PreprocessorBase):
    """
    Comprehensive preprocessor for multi-omics data.
    
    This class handles preprocessing of different omics data types including:
    - Genomics data (mutations, CNV)
    - Transcriptomics data (gene expression)
    - Drug sensitivity data
    - Clinical data
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize multi-omics preprocessor.
        
        Args:
            config: Configuration object containing preprocessing settings
            
        Raises:
            PreprocessingError: If initialization fails
        """
        try:
            self.config = config or load_config()
            self.is_fitted = False
            self.scalers = {}
            self.imputers = {}
            self.feature_selectors = {}
            self.preprocessing_stats = {}
            
            # Drug response specific parameters
            self.ic50_log_transform = getattr(self.config.preprocessing, 'ic50_log_transform', True)
            self.ic50_outlier_threshold = getattr(self.config.preprocessing, 'ic50_outlier_threshold', 3.0)
            self.min_drug_cell_lines = getattr(self.config.preprocessing, 'min_drug_cell_lines', 50)
            self.expression_log_transform = getattr(self.config.preprocessing, 'expression_log_transform', True)
            self.mutation_binary_encoding = getattr(self.config.preprocessing, 'mutation_binary_encoding', True)
            
            # Storage for drug response specific stats
            self.drug_response_stats = {}
            
            logger.info("Initialized MultiOmicsPreprocessor with drug response capabilities")
        except Exception as e:
            raise PreprocessingError(f"Failed to initialize MultiOmicsPreprocessor: {str(e)}")
    
    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> 'MultiOmicsPreprocessor':
        """
        Fit preprocessor to multi-omics data.
        
        Args:
            data: Dictionary of DataFrames with omics data types as keys
            **kwargs: Additional fitting parameters
            
        Returns:
            MultiOmicsPreprocessor: Fitted preprocessor instance
            
        Raises:
            PreprocessingError: If fitting fails
        """
        try:
            logger.info("Fitting MultiOmicsPreprocessor...")
            
            for omics_type, df in data.items():
                logger.info(f"Fitting preprocessor for {omics_type} data...")
                self._fit_omics_type(omics_type, df, **kwargs)
            
            self.is_fitted = True
            logger.info("MultiOmicsPreprocessor fitting completed")
            
            return self
        except Exception as e:
            raise PreprocessingError(f"Failed to fit preprocessor: {str(e)}")
    
    def transform(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Transform multi-omics data using fitted preprocessor.
        
        Args:
            data: Dictionary of DataFrames with omics data types as keys
            **kwargs: Additional transformation parameters
            
        Returns:
            Dict[str, pd.DataFrame]: Transformed data
            
        Raises:
            PreprocessingError: If transformation fails
            DataValidationError: If preprocessor is not fitted
        """
        if not self.is_fitted:
            raise DataValidationError("Preprocessor must be fitted before transformation")
        
        try:
            logger.info("Transforming multi-omics data...")
            
            transformed_data = {}
            
            for omics_type, df in data.items():
                logger.info(f"Transforming {omics_type} data...")
                transformed_data[omics_type] = self._transform_omics_type(omics_type, df, **kwargs)
            
            logger.info("Multi-omics data transformation completed")
            
            return transformed_data
        except Exception as e:
            raise PreprocessingError(f"Failed to transform data: {str(e)}")
    
    def _fit_omics_type(self, omics_type: str, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit preprocessor for a specific omics data type.
        
        Args:
            omics_type: Type of omics data (e.g., 'genomics', 'transcriptomics')
            data: DataFrame containing the omics data
            **kwargs: Additional parameters
        """
        try:
            # Store preprocessing statistics
            self.preprocessing_stats[omics_type] = {
                'original_shape': data.shape,
                'missing_values': data.isnull().sum().sum(),
                'missing_rate': data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            }
            
            # Configure preprocessing based on omics type
            if omics_type in ['transcriptomics', 'gene_expression']:
                self._fit_transcriptomics(omics_type, data, **kwargs)
            elif omics_type in ['genomics', 'mutations', 'cnv']:
                self._fit_genomics(omics_type, data, **kwargs)
            elif omics_type in ['drug_sensitivity', 'drug_response']:
                self._fit_drug_data(omics_type, data, **kwargs)
            else:
                self._fit_generic(omics_type, data, **kwargs)
        except Exception as e:
            logger.error(f"Failed to fit {omics_type} preprocessor: {str(e)}")
            raise
    
    def _transform_omics_type(self, omics_type: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform a specific omics data type.
        
        Args:
            omics_type: Type of omics data
            data: DataFrame to transform
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Transformed data
        """
        try:
            transformed_data = data.copy()
            
            # Apply imputation if fitted
            if omics_type in self.imputers:
                numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    transformed_data[numeric_cols] = self.imputers[omics_type].transform(
                        transformed_data[numeric_cols]
                    )
            
            # Apply scaling if fitted
            if omics_type in self.scalers:
                numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    transformed_data[numeric_cols] = self.scalers[omics_type].transform(
                        transformed_data[numeric_cols]
                    )
            
            # Apply feature selection if fitted
            if omics_type in self.feature_selectors:
                numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_features = self.feature_selectors[omics_type].transform(
                        transformed_data[numeric_cols]
                    )
                    selected_feature_names = self.feature_selectors[omics_type].get_feature_names_out(
                        numeric_cols
                    )
                    
                    # Reconstruct DataFrame with selected features
                    transformed_data = pd.DataFrame(
                        selected_features,
                        index=transformed_data.index,
                        columns=selected_feature_names
                    )
            
            return transformed_data
        except Exception as e:
            logger.error(f"Failed to transform {omics_type} data: {str(e)}")
            raise
    
    def _fit_transcriptomics(self, omics_type: str, data: pd.DataFrame, **kwargs) -> None:
        """Fit preprocessor for transcriptomics data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning(f"No numeric data found for {omics_type}")
            return
        
        # Imputation for missing values
        self.imputers[omics_type] = KNNImputer(n_neighbors=5)
        self.imputers[omics_type].fit(numeric_data)
        
        # Log transformation and standardization for gene expression
        self.scalers[omics_type] = StandardScaler()
        # Apply log transformation first (add small constant to avoid log(0))
        log_data = np.log2(numeric_data + 1)
        self.scalers[omics_type].fit(log_data)
        
        # Feature selection based on variance
        self.feature_selectors[omics_type] = VarianceThreshold(threshold=0.1)
        self.feature_selectors[omics_type].fit(log_data)
        
        logger.info(f"Fitted transcriptomics preprocessor for {numeric_data.shape[1]} features")
    
    def _fit_genomics(self, omics_type: str, data: pd.DataFrame, **kwargs) -> None:
        """Fit preprocessor for genomics data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning(f"No numeric data found for {omics_type}")
            return
        
        # Simple imputation for genomics data (often binary/categorical)
        self.imputers[omics_type] = SimpleImputer(strategy='most_frequent')
        self.imputers[omics_type].fit(numeric_data)
        
        # Robust scaling for genomics data
        self.scalers[omics_type] = RobustScaler()
        self.scalers[omics_type].fit(numeric_data)
        
        # Feature selection based on variance
        self.feature_selectors[omics_type] = VarianceThreshold(threshold=0.01)
        self.feature_selectors[omics_type].fit(numeric_data)
        
        logger.info(f"Fitted genomics preprocessor for {numeric_data.shape[1]} features")
    
    def _fit_drug_data(self, omics_type: str, data: pd.DataFrame, **kwargs) -> None:
        """Fit preprocessor for drug sensitivity data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning(f"No numeric data found for {omics_type}")
            return
        
        # KNN imputation for drug sensitivity data
        self.imputers[omics_type] = KNNImputer(n_neighbors=3)
        self.imputers[omics_type].fit(numeric_data)
        
        # Standard scaling for drug sensitivity
        self.scalers[omics_type] = StandardScaler()
        self.scalers[omics_type].fit(numeric_data)
        
        logger.info(f"Fitted drug data preprocessor for {numeric_data.shape[1]} features")
    
    def _fit_generic(self, omics_type: str, data: pd.DataFrame, **kwargs) -> None:
        """Fit generic preprocessor for unknown data types."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning(f"No numeric data found for {omics_type}")
            return
        
        # Generic preprocessing
        self.imputers[omics_type] = SimpleImputer(strategy='median')
        self.imputers[omics_type].fit(numeric_data)
        
        self.scalers[omics_type] = StandardScaler()
        self.scalers[omics_type].fit(numeric_data)
        
        logger.info(f"Fitted generic preprocessor for {omics_type} with {numeric_data.shape[1]} features")
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics for all omics types.
        
        Returns:
            Dict[str, Any]: Preprocessing statistics
        """
        return self.preprocessing_stats.copy()
    
    def save_preprocessor(self, output_path: Union[str, Path]) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            output_path: Path to save the preprocessor
            
        Raises:
            PreprocessingError: If saving fails
        """
        try:
            import pickle
            
            output_path = Path(output_path)
            ensure_directory(output_path.parent)
            
            preprocessor_data = {
                'scalers': self.scalers,
                'imputers': self.imputers,
                'feature_selectors': self.feature_selectors,
                'preprocessing_stats': self.preprocessing_stats,
                'is_fitted': self.is_fitted
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(preprocessor_data, f)
            
            logger.info(f"Preprocessor saved to {output_path}")
        except Exception as e:
            raise PreprocessingError(f"Failed to save preprocessor: {str(e)}")
    
    def load_preprocessor(self, input_path: Union[str, Path]) -> 'MultiOmicsPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            input_path: Path to load the preprocessor from
            
        Returns:
            MultiOmicsPreprocessor: Loaded preprocessor instance
            
        Raises:
            PreprocessingError: If loading fails
        """
        try:
            import pickle
            
            with open(input_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
            
            self.scalers = preprocessor_data['scalers']
            self.imputers = preprocessor_data['imputers']
            self.feature_selectors = preprocessor_data['feature_selectors']
            self.preprocessing_stats = preprocessor_data['preprocessing_stats']
            self.is_fitted = preprocessor_data['is_fitted']
            
            logger.info(f"Preprocessor loaded from {input_path}")
            
            return self
        except Exception as e:
            raise PreprocessingError(f"Failed to load preprocessor: {str(e)}")
    
    def preprocess_drug_response_data(self,
                                    drug_sensitivity_data: pd.DataFrame,
                                    target_metric: str = 'LN_IC50',
                                    **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Specialized preprocessing for drug response data.
        
        Args:
            drug_sensitivity_data: DataFrame with drug sensitivity measurements
            target_metric: Target metric to predict ('LN_IC50', 'IC50_uM', 'AUC')
            **kwargs: Additional parameters
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        try:
            logger.info(f"Preprocessing drug response data with target: {target_metric}")
            
            # Validate required columns
            required_cols = ['COSMIC_ID', 'DRUG_NAME', target_metric]
            missing_cols = [col for col in required_cols if col not in drug_sensitivity_data.columns]
            if missing_cols:
                raise PreprocessingError(f"Missing required columns: {missing_cols}")
            
            # Remove drugs with insufficient cell lines
            drug_counts = drug_sensitivity_data['DRUG_NAME'].value_counts()
            valid_drugs = drug_counts[drug_counts >= self.min_drug_cell_lines].index
            
            logger.info(f"Filtering drugs: {len(valid_drugs)}/{len(drug_counts)} drugs have >= {self.min_drug_cell_lines} cell lines")
            
            filtered_data = drug_sensitivity_data[
                drug_sensitivity_data['DRUG_NAME'].isin(valid_drugs)
            ].copy()
            
            # Handle outliers in target variable
            target_values = filtered_data[target_metric].dropna()
            z_scores = np.abs(zscore(target_values))
            outlier_mask = z_scores > self.ic50_outlier_threshold
            
            if outlier_mask.sum() > 0:
                logger.info(f"Removing {outlier_mask.sum()} outliers from {target_metric}")
                filtered_data = filtered_data[~filtered_data.index.isin(target_values[outlier_mask].index)]
            
            # Create pivot table for drug response matrix
            response_matrix = filtered_data.pivot_table(
                index='COSMIC_ID',
                columns='DRUG_NAME',
                values=target_metric,
                aggfunc='mean'  # Handle duplicates by averaging
            )
            
            # Store drug response statistics
            self.drug_response_stats = {
                'n_drugs': len(valid_drugs),
                'n_cell_lines': len(response_matrix),
                'target_metric': target_metric,
                'missing_rate': response_matrix.isnull().sum().sum() / (response_matrix.shape[0] * response_matrix.shape[1]),
                'target_range': [target_values.min(), target_values.max()],
                'outliers_removed': outlier_mask.sum()
            }
            
            logger.info(f"Drug response matrix: {response_matrix.shape[0]} cell lines x {response_matrix.shape[1]} drugs")
            
            return response_matrix, filtered_data
            
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess drug response data: {str(e)}")
    
    def preprocess_gdsc_genomics(self,
                               mutations_data: pd.DataFrame,
                               cnv_data: pd.DataFrame,
                               **kwargs) -> pd.DataFrame:
        """
        Specialized preprocessing for GDSC genomics data.
        
        Args:
            mutations_data: DataFrame with mutation data
            cnv_data: DataFrame with copy number variation data
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Processed genomics features
        """
        try:
            logger.info("Preprocessing GDSC genomics data...")
            
            # Process mutations (binary encoding)
            if self.mutation_binary_encoding:
                # Mutations are already binary in our format
                mutations_processed = mutations_data.set_index('COSMIC_ID')
                mutations_processed = mutations_processed.fillna(0).astype(int)
                
                # Add mutation burden (total mutations per cell line)
                mutations_processed['MUTATION_BURDEN'] = mutations_processed.sum(axis=1)
                
                logger.info(f"Processed mutations: {mutations_processed.shape}")
            
            # Process CNV data
            cnv_processed = cnv_data.set_index('COSMIC_ID')
            cnv_processed = cnv_processed.fillna(0)
            
            # Add CNV burden metrics
            cnv_processed['CNV_AMPLIFICATIONS'] = (cnv_processed > 0).sum(axis=1)
            cnv_processed['CNV_DELETIONS'] = (cnv_processed < 0).sum(axis=1)
            cnv_processed['CNV_BURDEN'] = cnv_processed.abs().sum(axis=1)
            
            logger.info(f"Processed CNV: {cnv_processed.shape}")
            
            # Combine genomics data
            genomics_combined = pd.concat([mutations_processed, cnv_processed], axis=1, sort=False)
            genomics_combined = genomics_combined.fillna(0)
            
            # Feature selection based on variance
            variance_selector = VarianceThreshold(threshold=0.01)
            selected_features = variance_selector.fit_transform(genomics_combined)
            selected_feature_names = genomics_combined.columns[variance_selector.get_support()]
            
            genomics_final = pd.DataFrame(
                selected_features,
                index=genomics_combined.index,
                columns=selected_feature_names
            )
            
            logger.info(f"Final genomics features: {genomics_final.shape}")
            
            return genomics_final
            
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess genomics data: {str(e)}")
    
    def preprocess_gdsc_expression(self,
                                 expression_data: pd.DataFrame,
                                 **kwargs) -> pd.DataFrame:
        """
        Specialized preprocessing for GDSC gene expression data.
        
        Args:
            expression_data: DataFrame with gene expression data
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Processed expression features
        """
        try:
            logger.info("Preprocessing GDSC gene expression data...")
            
            expression_processed = expression_data.set_index('COSMIC_ID')
            
            # Log transformation if specified
            if self.expression_log_transform:
                # Data is already log2 transformed in our simulation, but ensure positive values
                expression_processed = expression_processed.clip(lower=0)
                logger.info("Applied log transformation to expression data")
            
            # Remove low-variance genes
            variance_selector = VarianceThreshold(threshold=0.1)
            selected_features = variance_selector.fit_transform(expression_processed)
            selected_genes = expression_processed.columns[variance_selector.get_support()]
            
            expression_final = pd.DataFrame(
                selected_features,
                index=expression_processed.index,
                columns=selected_genes
            )
            
            # Z-score normalization
            expression_final = pd.DataFrame(
                zscore(expression_final, axis=0),
                index=expression_final.index,
                columns=expression_final.columns
            )
            
            logger.info(f"Final expression features: {expression_final.shape}")
            
            return expression_final
            
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess expression data: {str(e)}")
    
    def integrate_multiomics_for_drug_response(self,
                                             drug_response_matrix: pd.DataFrame,
                                             genomics_data: pd.DataFrame,
                                             expression_data: pd.DataFrame,
                                             target_drug: str,
                                             **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Integrate multi-omics data for drug response prediction.
        
        Args:
            drug_response_matrix: Drug response data matrix
            genomics_data: Processed genomics features
            expression_data: Processed expression features
            target_drug: Drug to predict response for
            **kwargs: Additional parameters
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Integrated features and target
        """
        try:
            logger.info(f"Integrating multi-omics data for drug: {target_drug}")
            
            # Get target drug response
            if target_drug not in drug_response_matrix.columns:
                raise PreprocessingError(f"Drug {target_drug} not found in response matrix")
            
            target_response = drug_response_matrix[target_drug].dropna()
            
            # Find common cell lines across all data types
            common_cell_lines = set(target_response.index)
            common_cell_lines = common_cell_lines.intersection(set(genomics_data.index))
            common_cell_lines = common_cell_lines.intersection(set(expression_data.index))
            
            if len(common_cell_lines) < 20:
                raise PreprocessingError(f"Insufficient common cell lines: {len(common_cell_lines)}")
            
            common_cell_lines = sorted(list(common_cell_lines))
            logger.info(f"Common cell lines: {len(common_cell_lines)}")
            
            # Subset data to common cell lines
            target_subset = target_response.loc[common_cell_lines]
            genomics_subset = genomics_data.loc[common_cell_lines]
            expression_subset = expression_data.loc[common_cell_lines]
            
            # Combine features with prefixes
            genomics_subset.columns = [f"genomics_{col}" for col in genomics_subset.columns]
            expression_subset.columns = [f"expression_{col}" for col in expression_subset.columns]
            
            integrated_features = pd.concat([genomics_subset, expression_subset], axis=1)
            
            # Feature selection based on correlation with target
            correlations = []
            for col in integrated_features.columns:
                try:
                    corr, p_val = pearsonr(integrated_features[col], target_subset)
                    correlations.append((col, abs(corr), p_val))
                except:
                    correlations.append((col, 0, 1))
            
            # Sort by correlation and select top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [item[0] for item in correlations[:kwargs.get('max_features', 1000)]]
            
            final_features = integrated_features[top_features]
            
            logger.info(f"Final integrated dataset: {final_features.shape[0]} samples x {final_features.shape[1]} features")
            
            return final_features, target_subset
            
        except Exception as e:
            raise PreprocessingError(f"Failed to integrate multi-omics data: {str(e)}")
    
    def get_drug_response_stats(self) -> Dict[str, Any]:
        """Get drug response preprocessing statistics."""
        return self.drug_response_stats.copy()


__all__ = ["MultiOmicsPreprocessor"]