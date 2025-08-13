"""
Extended model configuration for Multi-omics Biomarker Discovery

This module provides extended configuration classes for machine learning
models and hyperparameter optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ModelConfig:
    """Extended configuration for machine learning models"""
    
    # Algorithm Selection
    algorithms: List[str] = field(default_factory=lambda: [
        "ridge", "lasso", "elastic_net", "random_forest", 
        "xgboost", "lightgbm", "neural_network"
    ])
    
    # Cross-validation Configuration
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # Hyperparameter Optimization
    optuna_n_trials: int = 100
    optuna_timeout: int = 3600
    optuna_sampler: str = "TPE"  # TPE, Random, Grid
    
    # Model Performance Thresholds
    min_r2_score: float = 0.3
    min_auc_score: float = 0.7
    min_precision: float = 0.6
    min_recall: float = 0.6
    
    # Early Stopping
    early_stopping_rounds: int = 50
    early_stopping_metric: str = "rmse"
    
    # Model Ensemble Configuration
    ensemble_methods: List[str] = field(default_factory=lambda: [
        "voting", "stacking", "bagging"
    ])
    ensemble_weights: Optional[List[float]] = None
    
    def get_algorithm_params(self) -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameter ranges for each algorithm"""
        return {
            "ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0],
                "fit_intercept": [True, False],
                "normalize": [True, False]
            },
            "lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "fit_intercept": [True, False],
                "normalize": [True, False]
            },
            "elastic_net": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "fit_intercept": [True, False]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 100]
            },
            "neural_network": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            }
        }
    
    def get_evaluation_metrics(self) -> List[str]:
        """Get list of evaluation metrics to compute"""
        return [
            "r2_score", "mean_squared_error", "mean_absolute_error",
            "explained_variance_score", "max_error"
        ]
    
    def get_feature_selection_params(self) -> Dict[str, Any]:
        """Get feature selection parameters"""
        return {
            "methods": ["univariate", "recursive", "lasso", "tree_based"],
            "k_best": [10, 50, 100, 500],
            "percentile": [10, 25, 50, 75],
            "alpha": [0.001, 0.01, 0.1]
        }


__all__ = ["ModelConfig"]