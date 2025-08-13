"""
Biomarker Discovery Module for Multi-omics Data

This module provides comprehensive biomarker discovery functionality including:
- Feature selection and ranking methods
- Stability selection for robust biomarker identification
- Multi-omics biomarker integration
- Clinical relevance assessment
- Biomarker validation and interpretation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from abc import ABC, abstractmethod
import warnings

# Statistical and ML imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ..utils import setup_logging, load_config, ensure_directory


logger = logging.getLogger(__name__)


class BiomarkerDiscoveryBase(ABC):
    """Abstract base class for biomarker discovery methods."""
    
    @abstractmethod
    def discover_biomarkers(self, 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           **kwargs) -> pd.DataFrame:
        """Discover biomarkers from multi-omics data."""
        pass
    
    @abstractmethod
    def validate_biomarkers(self, 
                           biomarkers: List[str], 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           **kwargs) -> Dict[str, Any]:
        """Validate discovered biomarkers."""
        pass


class BiomarkerDiscovery(BiomarkerDiscoveryBase):
    """
    Comprehensive biomarker discovery system.
    
    This class provides multiple methods for discovering and validating biomarkers
    from multi-omics data, with emphasis on stability and clinical relevance.
    """
    
    def __init__(self, 
                 feature_selection_method: str = "stability_selection",
                 config: Optional[Any] = None):
        """
        Initialize biomarker discovery system.
        
        Args:
            feature_selection_method: Method for feature selection
            config: Configuration object
        """
        self.config = config or load_config()
        self.feature_selection_method = feature_selection_method
        
        # Configuration parameters
        self.feature_selection_threshold = getattr(
            self.config.biomarker, 'feature_selection_threshold', 0.6
        )
        self.max_features = getattr(self.config.biomarker, 'max_features', 1000)
        self.validation_folds = getattr(self.config.biomarker, 'validation_folds', 10)
        self.stability_threshold = getattr(self.config.biomarker, 'stability_threshold', 0.7)
        self.clinical_relevance_threshold = getattr(
            self.config.biomarker, 'clinical_relevance_threshold', 0.05
        )
        
        # Storage for results
        self.discovered_biomarkers = {}
        self.validation_results = {}
        self.feature_importance_scores = {}
        
        logger.info(f"Initialized BiomarkerDiscovery with method: {feature_selection_method}")
    
    def discover_biomarkers(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           omics_types: Optional[Dict[str, List[str]]] = None,
                           **kwargs) -> pd.DataFrame:
        """
        Discover biomarkers from multi-omics data.
        
        Args:
            X: Feature matrix (samples x features)
            y: Target variable (drug response)
            omics_types: Dictionary mapping omics types to feature names
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Discovered biomarkers with scores and metadata
        """
        logger.info("Starting biomarker discovery...")
        
        # Validate input data
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Detect omics types if not provided
        if omics_types is None:
            omics_types = self._detect_omics_types(X.columns)
        
        # Run feature selection based on method
        if self.feature_selection_method == "stability_selection":
            biomarkers = self._stability_selection(X, y, **kwargs)
        elif self.feature_selection_method == "univariate":
            biomarkers = self._univariate_selection(X, y, **kwargs)
        elif self.feature_selection_method == "multivariate":
            biomarkers = self._multivariate_selection(X, y, **kwargs)
        elif self.feature_selection_method == "ensemble":
            biomarkers = self._ensemble_selection(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        # Add omics type information
        biomarkers = self._add_omics_annotations(biomarkers, omics_types)
        
        # Calculate additional metrics
        biomarkers = self._calculate_biomarker_metrics(biomarkers, X, y)
        
        # Store results
        self.discovered_biomarkers = biomarkers
        
        logger.info(f"Discovered {len(biomarkers)} potential biomarkers")
        
        return biomarkers
    
    def _detect_omics_types(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Detect omics types from feature names."""
        omics_types = {
            "genomics": [],
            "transcriptomics": [],
            "drug_sensitivity": [],
            "unknown": []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ["mut", "cnv", "genomics"]):
                omics_types["genomics"].append(feature)
            elif any(keyword in feature_lower for keyword in ["expr", "gene", "transcriptomics"]):
                omics_types["transcriptomics"].append(feature)
            elif any(keyword in feature_lower for keyword in ["drug", "ic50", "sensitivity"]):
                omics_types["drug_sensitivity"].append(feature)
            else:
                omics_types["unknown"].append(feature)
        
        # Remove empty categories
        omics_types = {k: v for k, v in omics_types.items() if v}
        
        return omics_types
    
    def _stability_selection(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """Perform stability selection for robust biomarker discovery."""
        logger.info("Running stability selection...")
        
        n_bootstrap = kwargs.get("n_bootstrap", 100)
        subsample_ratio = kwargs.get("subsample_ratio", 0.8)
        
        feature_selection_counts = defaultdict(int)
        feature_importance_sums = defaultdict(float)
        
        # Bootstrap sampling and feature selection
        for i in range(n_bootstrap):
            # Subsample data
            n_samples = int(len(X) * subsample_ratio)
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            
            X_sub = X.iloc[sample_indices]
            y_sub = y.iloc[sample_indices]
            
            # Feature selection using Lasso
            lasso = LassoCV(cv=5, random_state=i)
            lasso.fit(X_sub, y_sub)
            
            # Get selected features
            selected_features = X_sub.columns[lasso.coef_ != 0]
            
            # Count selections
            for feature in selected_features:
                feature_selection_counts[feature] += 1
                feature_importance_sums[feature] += abs(lasso.coef_[X_sub.columns.get_loc(feature)])
        
        # Calculate stability scores
        biomarkers_data = []
        
        for feature, count in feature_selection_counts.items():
            stability_score = count / n_bootstrap
            
            if stability_score >= self.stability_threshold:
                avg_importance = feature_importance_sums[feature] / count
                
                biomarkers_data.append({
                    "feature": feature,
                    "stability_score": stability_score,
                    "avg_importance": avg_importance,
                    "selection_count": count,
                    "method": "stability_selection"
                })
        
        # Convert to DataFrame and sort
        biomarkers_df = pd.DataFrame(biomarkers_data)
        
        if not biomarkers_df.empty:
            biomarkers_df = biomarkers_df.sort_values("stability_score", ascending=False)
            
            # Limit to max_features
            if len(biomarkers_df) > self.max_features:
                biomarkers_df = biomarkers_df.head(self.max_features)
        
        return biomarkers_df
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """Perform univariate feature selection."""
        logger.info("Running univariate selection...")
        
        biomarkers_data = []
        
        # Calculate correlation and statistical significance for each feature
        for feature in X.columns:
            feature_values = X[feature].dropna()
            target_values = y[feature_values.index]
            
            if len(feature_values) < 10:  # Skip features with too few values
                continue
            
            # Pearson correlation
            corr_coef, p_value = pearsonr(feature_values, target_values)
            
            # F-statistic
            f_stat, f_p_value = stats.f_oneway(feature_values, target_values)
            
            biomarkers_data.append({
                "feature": feature,
                "correlation": abs(corr_coef),
                "correlation_p_value": p_value,
                "f_statistic": f_stat,
                "f_p_value": f_p_value,
                "method": "univariate"
            })
        
        # Convert to DataFrame
        biomarkers_df = pd.DataFrame(biomarkers_data)
        
        if not biomarkers_df.empty:
            # Multiple testing correction
            biomarkers_df["correlation_fdr"] = multipletests(
                biomarkers_df["correlation_p_value"], method="fdr_bh"
            )[1]
            
            biomarkers_df["f_fdr"] = multipletests(
                biomarkers_df["f_p_value"], method="fdr_bh"
            )[1]
            
            # Filter significant features
            significant_features = biomarkers_df[
                (biomarkers_df["correlation_fdr"] < 0.05) |
                (biomarkers_df["f_fdr"] < 0.05)
            ]
            
            # Sort by correlation strength
            significant_features = significant_features.sort_values("correlation", ascending=False)
            
            # Limit to max_features
            if len(significant_features) > self.max_features:
                significant_features = significant_features.head(self.max_features)
            
            return significant_features
        
        return biomarkers_df
    
    def _multivariate_selection(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """Perform multivariate feature selection."""
        logger.info("Running multivariate selection...")
        
        # Use ElasticNet for multivariate selection
        elastic_net = ElasticNetCV(cv=5, random_state=42)
        elastic_net.fit(X, y)
        
        # Get feature coefficients
        feature_coefs = elastic_net.coef_
        selected_features = X.columns[feature_coefs != 0]
        
        biomarkers_data = []
        
        for feature in selected_features:
            feature_idx = X.columns.get_loc(feature)
            coef_value = feature_coefs[feature_idx]
            
            biomarkers_data.append({
                "feature": feature,
                "coefficient": coef_value,
                "abs_coefficient": abs(coef_value),
                "method": "multivariate"
            })
        
        # Convert to DataFrame and sort
        biomarkers_df = pd.DataFrame(biomarkers_data)
        
        if not biomarkers_df.empty:
            biomarkers_df = biomarkers_df.sort_values("abs_coefficient", ascending=False)
            
            # Limit to max_features
            if len(biomarkers_df) > self.max_features:
                biomarkers_df = biomarkers_df.head(self.max_features)
        
        return biomarkers_df
    
    def _ensemble_selection(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> pd.DataFrame:
        """Perform ensemble feature selection combining multiple methods."""
        logger.info("Running ensemble selection...")
        
        # Run multiple selection methods
        stability_results = self._stability_selection(X, y, **kwargs)
        univariate_results = self._univariate_selection(X, y, **kwargs)
        multivariate_results = self._multivariate_selection(X, y, **kwargs)
        
        # Combine results
        all_features = set()
        if not stability_results.empty:
            all_features.update(stability_results["feature"])
        if not univariate_results.empty:
            all_features.update(univariate_results["feature"])
        if not multivariate_results.empty:
            all_features.update(multivariate_results["feature"])
        
        # Calculate ensemble scores
        biomarkers_data = []
        
        for feature in all_features:
            ensemble_score = 0
            method_count = 0
            
            # Stability selection score
            if not stability_results.empty and feature in stability_results["feature"].values:
                stability_score = stability_results[
                    stability_results["feature"] == feature
                ]["stability_score"].iloc[0]
                ensemble_score += stability_score
                method_count += 1
            
            # Univariate score (normalized correlation)
            if not univariate_results.empty and feature in univariate_results["feature"].values:
                corr_score = univariate_results[
                    univariate_results["feature"] == feature
                ]["correlation"].iloc[0]
                ensemble_score += corr_score
                method_count += 1
            
            # Multivariate score (normalized coefficient)
            if not multivariate_results.empty and feature in multivariate_results["feature"].values:
                coef_score = multivariate_results[
                    multivariate_results["feature"] == feature
                ]["abs_coefficient"].iloc[0]
                # Normalize coefficient (simple min-max normalization)
                max_coef = multivariate_results["abs_coefficient"].max()
                if max_coef > 0:
                    coef_score = coef_score / max_coef
                ensemble_score += coef_score
                method_count += 1
            
            # Average ensemble score
            if method_count > 0:
                ensemble_score = ensemble_score / method_count
                
                biomarkers_data.append({
                    "feature": feature,
                    "ensemble_score": ensemble_score,
                    "method_count": method_count,
                    "method": "ensemble"
                })
        
        # Convert to DataFrame and sort
        biomarkers_df = pd.DataFrame(biomarkers_data)
        
        if not biomarkers_df.empty:
            biomarkers_df = biomarkers_df.sort_values("ensemble_score", ascending=False)
            
            # Limit to max_features
            if len(biomarkers_df) > self.max_features:
                biomarkers_df = biomarkers_df.head(self.max_features)
        
        return biomarkers_df
    
    def _add_omics_annotations(self, 
                              biomarkers: pd.DataFrame, 
                              omics_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Add omics type annotations to biomarkers."""
        if biomarkers.empty:
            return biomarkers
        
        # Create omics type mapping
        feature_to_omics = {}
        for omics_type, features in omics_types.items():
            for feature in features:
                feature_to_omics[feature] = omics_type
        
        # Add omics type column
        biomarkers["omics_type"] = biomarkers["feature"].map(
            lambda x: feature_to_omics.get(x, "unknown")
        )
        
        return biomarkers
    
    def _calculate_biomarker_metrics(self, 
                                   biomarkers: pd.DataFrame, 
                                   X: pd.DataFrame, 
                                   y: pd.Series) -> pd.DataFrame:
        """Calculate additional metrics for biomarkers."""
        if biomarkers.empty:
            return biomarkers
        
        # Calculate additional metrics for each biomarker
        additional_metrics = []
        
        for _, biomarker in biomarkers.iterrows():
            feature = biomarker["feature"]
            
            if feature not in X.columns:
                continue
            
            feature_values = X[feature].dropna()
            target_values = y[feature_values.index]
            
            # Calculate various metrics
            metrics = {
                "mean_value": feature_values.mean(),
                "std_value": feature_values.std(),
                "missing_rate": X[feature].isnull().sum() / len(X),
                "variance": feature_values.var()
            }
            
            # Correlation with target
            if len(feature_values) > 1:
                corr_coef, corr_p = pearsonr(feature_values, target_values)
                metrics["target_correlation"] = corr_coef
                metrics["target_correlation_p"] = corr_p
            
            additional_metrics.append(metrics)
        
        # Add metrics to biomarkers DataFrame
        metrics_df = pd.DataFrame(additional_metrics)
        biomarkers_with_metrics = pd.concat([biomarkers.reset_index(drop=True), metrics_df], axis=1)
        
        return biomarkers_with_metrics
    
    def validate_biomarkers(self, 
                           biomarkers: Optional[List[str]] = None, 
                           X: pd.DataFrame = None, 
                           y: pd.Series = None,
                           validation_method: str = "cross_validation",
                           **kwargs) -> Dict[str, Any]:
        """
        Validate discovered biomarkers.
        
        Args:
            biomarkers: List of biomarker features to validate
            X: Feature matrix for validation
            y: Target variable for validation
            validation_method: Validation method to use
            **kwargs: Additional validation parameters
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info("Validating biomarkers...")
        
        # Use discovered biomarkers if not provided
        if biomarkers is None:
            if self.discovered_biomarkers.empty:
                raise ValueError("No biomarkers to validate. Run discover_biomarkers first.")
            biomarkers = self.discovered_biomarkers["feature"].tolist()
        
        if X is None or y is None:
            raise ValueError("Validation data (X, y) must be provided")
        
        # Filter to available biomarkers
        available_biomarkers = [b for b in biomarkers if b in X.columns]
        
        if not available_biomarkers:
            raise ValueError("None of the biomarkers are available in the validation data")
        
        # Run validation based on method
        if validation_method == "cross_validation":
            validation_results = self._cross_validation(available_biomarkers, X, y, **kwargs)
        elif validation_method == "bootstrap":
            validation_results = self._bootstrap_validation(available_biomarkers, X, y, **kwargs)
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
        
        # Store results
        self.validation_results = validation_results
        
        logger.info("Biomarker validation completed")
        
        return validation_results
    
    def _cross_validation(self, 
                         biomarkers: List[str], 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         **kwargs) -> Dict[str, Any]:
        """Perform cross-validation of biomarkers."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        
        # Use only biomarker features
        X_biomarkers = X[biomarkers]
        
        # Cross-validation with Ridge regression
        model = Ridge(alpha=1.0)
        cv_scores = cross_val_score(
            model, X_biomarkers, y, 
            cv=self.validation_folds, 
            scoring='r2'
        )
        
        validation_results = {
            "method": "cross_validation",
            "n_biomarkers": len(biomarkers),
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
            "biomarkers": biomarkers
        }
        
        return validation_results
    
    def _bootstrap_validation(self, 
                            biomarkers: List[str], 
                            X: pd.DataFrame, 
                            y: pd.Series, 
                            **kwargs) -> Dict[str, Any]:
        """Perform bootstrap validation of biomarkers."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        n_bootstrap = kwargs.get("n_bootstrap", 100)
        bootstrap_scores = []
        
        # Use only biomarker features
        X_biomarkers = X[biomarkers]
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X_biomarkers.iloc[sample_indices]
            y_boot = y.iloc[sample_indices]
            
            # Train and evaluate
            model = Ridge(alpha=1.0)
            model.fit(X_boot, y_boot)
            
            # Out-of-bag evaluation
            oob_indices = list(set(range(len(X))) - set(sample_indices))
            if oob_indices:
                X_oob = X_biomarkers.iloc[oob_indices]
                y_oob = y.iloc[oob_indices]
                
                y_pred = model.predict(X_oob)
                score = r2_score(y_oob, y_pred)
                bootstrap_scores.append(score)
        
        validation_results = {
            "method": "bootstrap",
            "n_biomarkers": len(biomarkers),
            "bootstrap_scores": bootstrap_scores,
            "mean_bootstrap_score": np.mean(bootstrap_scores),
            "std_bootstrap_score": np.std(bootstrap_scores),
            "biomarkers": biomarkers
        }
        
        return validation_results
    
    def get_top_biomarkers(self, n_top: int = 10) -> pd.DataFrame:
        """
        Get top N biomarkers.
        
        Args:
            n_top: Number of top biomarkers to return
            
        Returns:
            pd.DataFrame: Top biomarkers
        """
        if self.discovered_biomarkers.empty:
            raise ValueError("No biomarkers discovered yet. Run discover_biomarkers first.")
        
        return self.discovered_biomarkers.head(n_top)
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save biomarker discovery results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        ensure_directory(output_dir)
        
        # Save discovered biomarkers
        if not self.discovered_biomarkers.empty:
            biomarkers_path = output_dir / "discovered_biomarkers.csv"
            self.discovered_biomarkers.to_csv(biomarkers_path, index=False)
            logger.info(f"Saved biomarkers to {biomarkers_path}")
        
        # Save validation results
        if self.validation_results:
            import json
            validation_path = output_dir / "validation_results.json"
            
            with open(validation_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"Saved validation results to {validation_path}")


# Import defaultdict for collections
from collections import defaultdict

# Export main classes
__all__ = [
    "BiomarkerDiscovery",
    "BiomarkerDiscoveryBase"
]