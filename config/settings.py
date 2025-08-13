"""
Main configuration settings for Multi-omics Biomarker Discovery

This module provides the core configuration management system with support for:
- Hierarchical configuration loading
- Environment variable overrides
- Type validation with Pydantic
- Multiple environment support (dev/test/prod)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data acquisition and storage configuration"""
    gdsc_version: str = "v2.0"
    cancer_types: List[str] = field(default_factory=lambda: ["BRCA", "LUAD"])
    drugs: List[str] = field(default_factory=lambda: ["Paclitaxel", "Cisplatin"])
    min_samples_per_drug: int = 10
    max_missing_rate: float = 0.2
    
    # Data paths
    raw_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"
    results_path: str = "./data/results"
    
    # External data sources
    gdsc_base_url: str = "https://www.cancerrxgene.org/downloads/bulk_download"
    kegg_api_url: str = "https://rest.kegg.jp"
    reactome_api_url: str = "https://reactome.org/ContentService"
    go_api_url: str = "http://api.geneontology.org"
    ensembl_api_url: str = "https://rest.ensembl.org"

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    algorithms: List[str] = field(default_factory=lambda: [
        "ridge", "lasso", "elastic_net", "random_forest", 
        "xgboost", "lightgbm", "neural_network"
    ])
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # Hyperparameter optimization
    optuna_n_trials: int = 100
    optuna_timeout: int = 3600
    
    # Model performance thresholds
    min_r2_score: float = 0.3
    min_auc_score: float = 0.7

@dataclass
class PathwayConfig:
    """Pathway analysis configuration"""
    databases: List[str] = field(default_factory=lambda: [
        "KEGG", "Reactome", "GO", "MSigDB"
    ])
    p_value_threshold: float = 0.05
    fdr_threshold: float = 0.1
    min_pathway_size: int = 5
    max_pathway_size: int = 500
    
    # Network analysis
    network_edge_threshold: float = 0.5
    network_min_degree: int = 2

@dataclass
class BiomarkerConfig:
    """Biomarker discovery configuration"""
    feature_selection_method: str = "stability_selection"
    feature_selection_threshold: float = 0.6
    max_features: int = 1000
    
    # Validation
    validation_folds: int = 10
    stability_threshold: float = 0.7
    clinical_relevance_threshold: float = 0.05

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "./logs/multiomics.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class DatabaseConfig:
    """Database configuration"""
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "multiomics"
    postgres_user: str = "researcher"
    postgres_password: str = "secure_password_2024"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "redis_password_2024"
    redis_db: int = 0
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "multiomics-biomarker-discovery"
    mlflow_artifact_root: str = "./mlruns"
    
    # Weights & Biases
    wandb_project: str = "multiomics-biomarker-discovery"
    wandb_entity: str = "your_wandb_entity"
    wandb_mode: str = "online"  # online, offline, disabled

@dataclass
class ProjectConfig:
    """Main project configuration"""
    # Project metadata
    project_name: str = "multiomics-biomarker-discovery"
    version: str = "1.0.0"
    environment: str = "development"  # development, testing, production
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)
    biomarker: BiomarkerConfig = field(default_factory=BiomarkerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectConfig':
        """Create configuration from dictionary"""
        # Apply environment variable overrides
        config_dict = cls._apply_env_overrides(config_dict)
        
        # Create sub-configurations
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        pathway_config = PathwayConfig(**config_dict.get('pathway', {}))
        biomarker_config = BiomarkerConfig(**config_dict.get('biomarker', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        database_config = DatabaseConfig(**config_dict.get('database', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
        
        return cls(
            project_name=config_dict.get('project_name', 'multiomics-biomarker-discovery'),
            version=config_dict.get('version', '1.0.0'),
            environment=config_dict.get('environment', 'development'),
            data=data_config,
            model=model_config,
            pathway=pathway_config,
            biomarker=biomarker_config,
            logging=logging_config,
            database=database_config,
            experiment=experiment_config
        )
    
    @staticmethod
    def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        # Environment variable mapping
        env_mappings = {
            'GDSC_VERSION': ['data', 'gdsc_version'],
            'GDSC_BASE_URL': ['data', 'gdsc_base_url'],
            'ML_RANDOM_STATE': ['model', 'random_state'],
            'ML_CV_FOLDS': ['model', 'cv_folds'],
            'PATHWAY_P_VALUE_THRESHOLD': ['pathway', 'p_value_threshold'],
            'LOG_LEVEL': ['logging', 'level'],
            'POSTGRES_HOST': ['database', 'postgres_host'],
            'POSTGRES_PORT': ['database', 'postgres_port'],
            'POSTGRES_DB': ['database', 'postgres_db'],
            'POSTGRES_USER': ['database', 'postgres_user'],
            'POSTGRES_PASSWORD': ['database', 'postgres_password'],
            'REDIS_HOST': ['database', 'redis_host'],
            'REDIS_PORT': ['database', 'redis_port'],
            'REDIS_PASSWORD': ['database', 'redis_password'],
            'MLFLOW_TRACKING_URI': ['experiment', 'mlflow_tracking_uri'],
            'WANDB_PROJECT': ['experiment', 'wandb_project'],
            'WANDB_ENTITY': ['experiment', 'wandb_entity'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested dictionary
                current_dict = config_dict
                for key in config_path[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                
                # Set the value with appropriate type conversion
                final_key = config_path[-1]
                if env_var.endswith('_PORT'):
                    current_dict[final_key] = int(env_value)
                elif env_var.endswith('_THRESHOLD') or env_var.endswith('_SIZE'):
                    current_dict[final_key] = float(env_value)
                elif env_var in ['ML_RANDOM_STATE', 'ML_CV_FOLDS']:
                    current_dict[final_key] = int(env_value)
                else:
                    current_dict[final_key] = env_value
        
        return config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'project_name': self.project_name,
            'version': self.version,
            'environment': self.environment,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'pathway': self.pathway.__dict__,
            'biomarker': self.biomarker.__dict__,
            'logging': self.logging.__dict__,
            'database': self.database.__dict__,
            'experiment': self.experiment.__dict__,
        }
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

class ConfigValidator(BaseModel):
    """Pydantic model for configuration validation"""
    project_name: str = Field(..., min_length=1)
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    environment: str = Field(..., regex=r'^(development|testing|production)$')
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v

def load_config(config_path: Optional[Union[str, Path]] = None) -> ProjectConfig:
    """
    Load project configuration from file or create default configuration
    
    Args:
        config_path: Path to configuration file. If None, uses default paths.
        
    Returns:
        ProjectConfig: Loaded configuration
    """
    if config_path is None:
        # Try default configuration paths
        default_paths = [
            Path("config/settings.yaml"),
            Path("config/config.yaml"),
            Path("settings.yaml"),
            Path("config.yaml")
        ]
        
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and Path(config_path).exists():
        return ProjectConfig.from_yaml(config_path)
    else:
        # Return default configuration
        return ProjectConfig()

def validate_config(config: ProjectConfig) -> bool:
    """
    Validate configuration using Pydantic
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if valid, raises ValidationError if invalid
    """
    try:
        ConfigValidator(
            project_name=config.project_name,
            version=config.version,
            environment=config.environment
        )
        return True
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

def get_config() -> ProjectConfig:
    """Get the global configuration instance"""
    if not hasattr(get_config, '_config'):
        get_config._config = load_config()
    return get_config._config

# Global configuration instance
CONFIG = get_config()