"""
Machine Learning Modeling Module for Drug Response Prediction

This module provides comprehensive machine learning capabilities for predicting drug response
from multi-omics data including:
- Multiple ML algorithms (linear, tree-based, neural networks)
- Hyperparameter optimization
- Cross-validation and model evaluation
- Model interpretation and feature importance
- Ensemble methods
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
import warnings

# Machine learning imports
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from ..utils import setup_logging, load_config, ensure_directory


logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """Abstract base class for drug response prediction models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ModelBase':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Evaluate model performance."""
        pass


class DrugResponsePredictor(ModelBase):
    """
    Comprehensive drug response prediction system.
    
    This class provides a unified interface for training and evaluating multiple
    machine learning models for drug response prediction from multi-omics data.
    """
    
    def __init__(self, 
                 algorithms: Optional[List[str]] = None,
                 config: Optional[Any] = None):
        """
        Initialize drug response predictor.
        
        Args:
            algorithms: List of algorithms to use. If None, uses config defaults.
            config: Configuration object
        """
        self.config = config or load_config()
        
        if algorithms is None:
            algorithms = self.config.model.algorithms if hasattr(self.config.model, 'algorithms') else [
                "ridge", "lasso", "random_forest"
            ]
        
        self.algorithms = algorithms
        self.models = {}
        self.best_model = None
        self.model_performances = {}
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Initialized DrugResponsePredictor with algorithms: {algorithms}")
    
    def _initialize_models(self) -> None:
        """Initialize machine learning models."""
        model_configs = {
            "ridge": {
                "model": Ridge(),
                "param_grid": {"alpha": [0.1, 1.0, 10.0, 100.0]}
            },
            "lasso": {
                "model": Lasso(),
                "param_grid": {"alpha": [0.01, 0.1, 1.0, 10.0]}
            },
            "elastic_net": {
                "model": ElasticNet(),
                "param_grid": {
                    "alpha": [0.01, 0.1, 1.0],
                    "l1_ratio": [0.1, 0.5, 0.9]
                }
            },
            "random_forest": {
                "model": RandomForestRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            },
            "neural_network": {
                "model": MLPRegressor(random_state=42, max_iter=1000),
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "alpha": [0.001, 0.01, 0.1],
                    "learning_rate_init": [0.001, 0.01]
                }
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST and "xgboost" in self.algorithms:
            model_configs["xgboost"] = {
                "model": xgb.XGBRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }
        
        # Add LightGBM if available
        if HAS_LIGHTGBM and "lightgbm" in self.algorithms:
            model_configs["lightgbm"] = {
                "model": lgb.LGBMRegressor(random_state=42, verbose=-1),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }
        
        # Store only requested algorithms
        for algorithm in self.algorithms:
            if algorithm in model_configs:
                self.models[algorithm] = model_configs[algorithm]
            else:
                logger.warning(f"Algorithm {algorithm} not available, skipping")
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            optimize_hyperparameters: bool = True,
            cv_folds: int = 5,
            **kwargs) -> 'DrugResponsePredictor':
        """
        Fit all models to training data.
        
        Args:
            X: Feature matrix
            y: Target variable (drug response)
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds
            **kwargs: Additional fitting parameters
            
        Returns:
            DrugResponsePredictor: Fitted predictor instance
        """
        logger.info("Fitting DrugResponsePredictor...")
        
        # Validate input data
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Fit each model
        for algorithm, model_config in self.models.items():
            logger.info(f"Fitting {algorithm} model...")
            
            try:
                if optimize_hyperparameters:
                    # Hyperparameter optimization
                    fitted_model = self._fit_with_optimization(
                        model_config, X, y, cv_folds, **kwargs
                    )
                else:
                    # Simple fitting without optimization
                    fitted_model = model_config["model"]
                    fitted_model.fit(X, y)
                
                self.models[algorithm]["fitted_model"] = fitted_model
                
                # Evaluate model performance
                performance = self._evaluate_model(fitted_model, X, y, cv_folds)
                self.model_performances[algorithm] = performance
                
                logger.info(f"{algorithm} - R² Score: {performance['r2_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {algorithm}: {str(e)}")
                continue
        
        # Select best model based on R² score
        if self.model_performances:
            best_algorithm = max(
                self.model_performances.keys(),
                key=lambda k: self.model_performances[k]['r2_score']
            )
            self.best_model = {
                "algorithm": best_algorithm,
                "model": self.models[best_algorithm]["fitted_model"],
                "performance": self.model_performances[best_algorithm]
            }
            
            logger.info(f"Best model: {best_algorithm} (R² = {self.best_model['performance']['r2_score']:.4f})")
        
        self.is_fitted = True
        logger.info("DrugResponsePredictor fitting completed")
        
        return self
    
    def _fit_with_optimization(self, 
                              model_config: Dict[str, Any], 
                              X: pd.DataFrame, 
                              y: pd.Series, 
                              cv_folds: int,
                              **kwargs) -> Any:
        """Fit model with hyperparameter optimization."""
        model = model_config["model"]
        param_grid = model_config["param_grid"]
        
        # Use Optuna if available and configured
        if HAS_OPTUNA and kwargs.get("use_optuna", False):
            return self._fit_with_optuna(model, param_grid, X, y, cv_folds)
        else:
            # Use GridSearchCV
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
    
    def _fit_with_optuna(self, 
                        model: Any, 
                        param_grid: Dict[str, List], 
                        X: pd.DataFrame, 
                        y: pd.Series, 
                        cv_folds: int) -> Any:
        """Fit model using Optuna optimization."""
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create model with sampled parameters
            model_instance = model.__class__(**params)
            
            # Cross-validation score
            scores = cross_val_score(model_instance, X, y, cv=cv_folds, scoring='r2')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Return best model
        best_model = model.__class__(**study.best_params)
        best_model.fit(X, y)
        
        return best_model
    
    def _evaluate_model(self, 
                       model: Any, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       cv_folds: int) -> Dict[str, float]:
        """Evaluate model performance using cross-validation."""
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        # Training set predictions for additional metrics
        y_pred = model.predict(X)
        
        performance = {
            'r2_score': cv_scores.mean(),
            'r2_std': cv_scores.std(),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return performance
    
    def predict(self, X: pd.DataFrame, use_best_model: bool = True, **kwargs) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            use_best_model: Whether to use the best model or ensemble
            **kwargs: Additional prediction parameters
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions")
        
        if use_best_model and self.best_model:
            return self.best_model["model"].predict(X)
        else:
            # Ensemble prediction (average of all models)
            predictions = []
            
            for algorithm, model_info in self.models.items():
                if "fitted_model" in model_info:
                    pred = model_info["fitted_model"].predict(X)
                    predictions.append(pred)
            
            if predictions:
                return np.mean(predictions, axis=0)
            else:
                raise ValueError("No fitted models available for prediction")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test feature matrix
            y: Test target variable
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before evaluation")
        
        predictions = self.predict(X, **kwargs)
        
        metrics = {
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Optional[pd.DataFrame]: Feature importance scores
        """
        if not self.best_model:
            return None
        
        model = self.best_model["model"]
        
        # Extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_scores = np.abs(model.coef_)
        else:
            logger.warning("Feature importance not available for this model type")
            return None
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_model_performances(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all fitted models.
        
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics by algorithm
        """
        return self.model_performances.copy()
    
    def save_models(self, output_dir: Union[str, Path]) -> None:
        """
        Save fitted models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import pickle
        
        output_dir = Path(output_dir)
        ensure_directory(output_dir)
        
        # Save each fitted model
        for algorithm, model_info in self.models.items():
            if "fitted_model" in model_info:
                model_path = output_dir / f"{algorithm}_model.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info["fitted_model"], f)
                
                logger.info(f"Saved {algorithm} model to {model_path}")
        
        # Save predictor metadata
        metadata = {
            'algorithms': self.algorithms,
            'model_performances': self.model_performances,
            'best_model': {
                'algorithm': self.best_model['algorithm'],
                'performance': self.best_model['performance']
            } if self.best_model else None
        }
        
        metadata_path = output_dir / "predictor_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved predictor metadata to {metadata_path}")


class DrugResponseEvaluator:
    """
    Specialized evaluator for drug response prediction models.
    
    This class provides drug response-specific evaluation metrics and validation
    methods including cross-validation, stratified evaluation, and clinical relevance metrics.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize drug response evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config or load_config()
        self.evaluation_results = {}
        
        logger.info("Initialized DrugResponseEvaluator")
    
    def evaluate_drug_response_model(self,
                                   model: Any,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   drug_name: str,
                                   **kwargs) -> Dict[str, float]:
        """
        Comprehensive evaluation of drug response prediction model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test drug response values
            drug_name: Name of the drug being evaluated
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, float]: Comprehensive evaluation metrics
        """
        try:
            logger.info(f"Evaluating drug response model for {drug_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Basic regression metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Drug response specific metrics
            metrics.update(self._calculate_drug_specific_metrics(y_test, y_pred, **kwargs))
            
            # Clinical relevance metrics
            metrics.update(self._calculate_clinical_metrics(y_test, y_pred, **kwargs))
            
            # Store results
            self.evaluation_results[drug_name] = metrics
            
            logger.info(f"Drug {drug_name} - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate drug response model: {str(e)}")
            raise
    
    def _calculate_drug_specific_metrics(self, y_true: pd.Series, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """Calculate drug response specific metrics."""
        metrics = {}
        
        try:
            # Pearson correlation
            from scipy.stats import pearsonr, spearmanr
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson_correlation'] = pearson_corr
            metrics['pearson_p_value'] = pearson_p
            
            # Spearman correlation (rank-based, more robust)
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
            
            # Prediction accuracy within tolerance
            tolerance = kwargs.get('tolerance', 0.5)  # Log units
            within_tolerance = np.abs(y_true - y_pred) <= tolerance
            metrics['accuracy_within_tolerance'] = within_tolerance.mean()
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate drug-specific metrics: {str(e)}")
            return {}
    
    def _calculate_clinical_metrics(self, y_true: pd.Series, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """Calculate clinically relevant metrics."""
        metrics = {}
        
        try:
            # Sensitive/Resistant classification based on thresholds
            sensitive_threshold = kwargs.get('sensitive_threshold', -1.0)  # LN_IC50 threshold
            resistant_threshold = kwargs.get('resistant_threshold', 1.0)
            
            # True classifications
            true_sensitive = y_true <= sensitive_threshold
            true_resistant = y_true >= resistant_threshold
            
            # Predicted classifications
            pred_sensitive = y_pred <= sensitive_threshold
            pred_resistant = y_pred >= resistant_threshold
            
            # Classification metrics for sensitive samples
            if true_sensitive.sum() > 0 and pred_sensitive.sum() > 0:
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # Convert to binary classification
                y_true_binary = (y_true <= sensitive_threshold).astype(int)
                y_pred_binary = (y_pred <= sensitive_threshold).astype(int)
                
                metrics['sensitive_precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                metrics['sensitive_recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                metrics['sensitive_f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate clinical metrics: {str(e)}")
            return {}
    
    def cross_validate_drug_response(self,
                                   predictor: DrugResponsePredictor,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   drug_name: str,
                                   cv_folds: int = 5,
                                   **kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation for drug response prediction.
        
        Args:
            predictor: Drug response predictor
            X: Feature matrix
            y: Target drug response
            drug_name: Name of the drug
            cv_folds: Number of CV folds
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        try:
            logger.info(f"Cross-validating drug response model for {drug_name}")
            
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_results = {
                'fold_scores': [],
                'fold_metrics': [],
                'mean_metrics': {},
                'std_metrics': {}
            }
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                logger.info(f"Processing fold {fold + 1}/{cv_folds}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                fold_predictor = DrugResponsePredictor(
                    algorithms=predictor.algorithms,
                    config=predictor.config
                )
                fold_predictor.fit(X_train, y_train, **kwargs)
                
                # Evaluate
                fold_metrics = self.evaluate_drug_response_model(
                    fold_predictor.best_model['model'],
                    X_test, y_test, f"{drug_name}_fold_{fold}"
                )
                
                cv_results['fold_scores'].append(fold_metrics['r2_score'])
                cv_results['fold_metrics'].append(fold_metrics)
            
            # Calculate mean and std across folds
            metrics_keys = cv_results['fold_metrics'][0].keys()
            for key in metrics_keys:
                values = [fold_metrics[key] for fold_metrics in cv_results['fold_metrics']]
                cv_results['mean_metrics'][key] = np.mean(values)
                cv_results['std_metrics'][key] = np.std(values)
            
            logger.info(f"CV Results for {drug_name}: R² = {cv_results['mean_metrics']['r2_score']:.4f} ± {cv_results['std_metrics']['r2_score']:.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise


class MultiDrugResponsePredictor:
    """
    Multi-drug response predictor for comprehensive drug sensitivity analysis.
    
    This class handles prediction across multiple drugs simultaneously and provides
    comparative analysis capabilities.
    """
    
    def __init__(self,
                 algorithms: Optional[List[str]] = None,
                 config: Optional[Any] = None):
        """
        Initialize multi-drug response predictor.
        
        Args:
            algorithms: List of algorithms to use
            config: Configuration object
        """
        self.config = config or load_config()
        self.algorithms = algorithms or ["ridge", "random_forest", "xgboost"]
        
        self.drug_predictors = {}
        self.drug_performances = {}
        self.evaluator = DrugResponseEvaluator(config)
        
        logger.info(f"Initialized MultiDrugResponsePredictor with {len(self.algorithms)} algorithms")
    
    def fit_multiple_drugs(self,
                          integrated_features: pd.DataFrame,
                          drug_response_matrix: pd.DataFrame,
                          drugs_to_model: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Fit models for multiple drugs.
        
        Args:
            integrated_features: Multi-omics integrated features
            drug_response_matrix: Drug response matrix (cell lines x drugs)
            drugs_to_model: List of drugs to model (if None, models all)
            **kwargs: Additional fitting parameters
            
        Returns:
            Dict[str, Any]: Fitting results summary
        """
        try:
            if drugs_to_model is None:
                drugs_to_model = drug_response_matrix.columns.tolist()
            
            logger.info(f"Fitting models for {len(drugs_to_model)} drugs")
            
            results_summary = {
                'successful_drugs': [],
                'failed_drugs': [],
                'performance_summary': {}
            }
            
            for drug in drugs_to_model:
                try:
                    logger.info(f"Fitting model for drug: {drug}")
                    
                    # Get drug response data
                    drug_response = drug_response_matrix[drug].dropna()
                    
                    # Find common samples
                    common_samples = list(set(integrated_features.index).intersection(set(drug_response.index)))
                    
                    if len(common_samples) < 50:
                        logger.warning(f"Insufficient samples for {drug}: {len(common_samples)}")
                        results_summary['failed_drugs'].append(drug)
                        continue
                    
                    # Subset data
                    X_drug = integrated_features.loc[common_samples]
                    y_drug = drug_response.loc[common_samples]
                    
                    # Train predictor
                    predictor = DrugResponsePredictor(
                        algorithms=self.algorithms,
                        config=self.config
                    )
                    predictor.fit(X_drug, y_drug, **kwargs)
                    
                    # Store predictor and performance
                    self.drug_predictors[drug] = predictor
                    self.drug_performances[drug] = predictor.get_model_performances()
                    
                    results_summary['successful_drugs'].append(drug)
                    results_summary['performance_summary'][drug] = {
                        'best_algorithm': predictor.best_model['algorithm'],
                        'best_r2': predictor.best_model['performance']['r2_score'],
                        'n_samples': len(common_samples)
                    }
                    
                    logger.info(f"Successfully fitted {drug}: R² = {predictor.best_model['performance']['r2_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to fit model for {drug}: {str(e)}")
                    results_summary['failed_drugs'].append(drug)
                    continue
            
            logger.info(f"Multi-drug fitting completed: {len(results_summary['successful_drugs'])}/{len(drugs_to_model)} successful")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Multi-drug fitting failed: {str(e)}")
            raise
    
    def predict_drug_response(self,
                            X_new: pd.DataFrame,
                            drugs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Predict drug response for new samples across multiple drugs.
        
        Args:
            X_new: New samples to predict
            drugs: List of drugs to predict (if None, predicts all fitted drugs)
            
        Returns:
            pd.DataFrame: Predictions matrix (samples x drugs)
        """
        try:
            if drugs is None:
                drugs = list(self.drug_predictors.keys())
            
            predictions = {}
            
            for drug in drugs:
                if drug in self.drug_predictors:
                    pred = self.drug_predictors[drug].predict(X_new)
                    predictions[drug] = pred
                else:
                    logger.warning(f"No fitted model found for drug: {drug}")
            
            predictions_df = pd.DataFrame(predictions, index=X_new.index)
            
            logger.info(f"Generated predictions for {len(predictions_df)} samples across {len(drugs)} drugs")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Multi-drug prediction failed: {str(e)}")
            raise
    
    def get_drug_rankings(self, sample_id: str, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get drug sensitivity rankings for a specific sample.
        
        Args:
            sample_id: Sample identifier
            predictions_df: Predictions DataFrame
            
        Returns:
            pd.DataFrame: Ranked drugs by predicted sensitivity
        """
        try:
            if sample_id not in predictions_df.index:
                raise ValueError(f"Sample {sample_id} not found in predictions")
            
            sample_predictions = predictions_df.loc[sample_id]
            
            # Sort by predicted response (lower = more sensitive for LN_IC50)
            rankings = sample_predictions.sort_values().reset_index()
            rankings.columns = ['Drug', 'Predicted_Response']
            rankings['Sensitivity_Rank'] = range(1, len(rankings) + 1)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to generate drug rankings: {str(e)}")
            raise


# Export main classes
__all__ = [
    "DrugResponsePredictor",
    "DrugResponseEvaluator",
    "MultiDrugResponsePredictor",
    "ModelBase"
]