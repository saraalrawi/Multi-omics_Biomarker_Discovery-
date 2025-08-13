"""
Data source configuration for Multi-omics Biomarker Discovery

This module provides configuration classes for external data sources
and API endpoints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataSourceConfig:
    """Configuration for external data sources and APIs"""
    
    # GDSC Configuration
    gdsc_version: str = "v2.0"
    gdsc_base_url: str = "https://www.cancerrxgene.org/downloads/bulk_download"
    
    # Pathway Database APIs
    kegg_api_url: str = "https://rest.kegg.jp"
    reactome_api_url: str = "https://reactome.org/ContentService"
    go_api_url: str = "http://api.geneontology.org"
    ensembl_api_url: str = "https://rest.ensembl.org"
    
    # MSigDB Configuration
    msigdb_url: str = "https://www.gsea-msigdb.org/gsea/msigdb"
    msigdb_collections: List[str] = field(default_factory=lambda: [
        "H", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"
    ])
    
    # API Rate Limiting
    api_rate_limit: int = 10  # requests per second
    api_timeout: int = 30  # seconds
    api_retry_attempts: int = 3
    
    # Cache Configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds (1 hour)
    cache_max_size: int = 1000  # maximum cached items
    
    # Data Quality Thresholds
    min_samples_per_drug: int = 10
    max_missing_rate: float = 0.2
    min_variance_threshold: float = 0.01
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get standard API headers for requests"""
        return {
            "User-Agent": "Multi-omics-Biomarker-Discovery/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def get_gdsc_endpoints(self) -> Dict[str, str]:
        """Get GDSC API endpoints"""
        return {
            "drug_response": f"{self.gdsc_base_url}/GDSC2_fitted_dose_response_25Feb20.xlsx",
            "drug_info": f"{self.gdsc_base_url}/screened_compunds_rel_8.2.csv",
            "cell_lines": f"{self.gdsc_base_url}/Cell_Lines_Details.xlsx",
            "genomics": f"{self.gdsc_base_url}/WES_variants.csv",
            "expression": f"{self.gdsc_base_url}/Cell_line_RMA_proc_basalExp.txt"
        }
    
    def get_pathway_endpoints(self) -> Dict[str, str]:
        """Get pathway database endpoints"""
        return {
            "kegg_pathways": f"{self.kegg_api_url}/list/pathway",
            "kegg_genes": f"{self.kegg_api_url}/list/genes",
            "reactome_pathways": f"{self.reactome_api_url}/data/pathways/low/diagram/entity",
            "go_terms": f"{self.go_api_url}/api/ontology/ribbon",
            "ensembl_genes": f"{self.ensembl_api_url}/lookup/genome/homo_sapiens"
        }


__all__ = ["DataSourceConfig"]