"""
Pathway Analysis Module for Multi-omics Biomarker Discovery

This module provides comprehensive pathway analysis functionality including:
- Gene set enrichment analysis (GSEA)
- Over-representation analysis (ORA)
- Pathway network analysis
- Integration with multiple pathway databases (KEGG, Reactome, GO, MSigDB)
- Functional annotation and interpretation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from abc import ABC, abstractmethod
import requests
import json
from urllib.parse import urljoin
import time
from collections import defaultdict

# Statistical analysis
from scipy import stats
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests

from ..utils import setup_logging, load_config, ensure_directory


logger = logging.getLogger(__name__)


class PathwayAnalysisBase(ABC):
    """Abstract base class for pathway analysis methods."""
    
    @abstractmethod
    def run_analysis(self, gene_list: List[str], **kwargs) -> pd.DataFrame:
        """Run pathway analysis on gene list."""
        pass
    
    @abstractmethod
    def get_pathways(self, **kwargs) -> Dict[str, List[str]]:
        """Get pathway gene sets."""
        pass


class PathwayAnalyzer(PathwayAnalysisBase):
    """
    Comprehensive pathway analysis system.
    
    This class provides multiple pathway analysis methods and integrates with
    various pathway databases to provide functional interpretation of gene lists.
    """
    
    def __init__(self, 
                 databases: Optional[List[str]] = None,
                 config: Optional[Any] = None):
        """
        Initialize pathway analyzer.
        
        Args:
            databases: List of pathway databases to use
            config: Configuration object
        """
        self.config = config or load_config()
        
        if databases is None:
            databases = self.config.pathway.databases if hasattr(self.config.pathway, 'databases') else [
                "KEGG", "Reactome", "GO"
            ]
        
        self.databases = databases
        self.pathway_data = {}
        self.analysis_results = {}
        
        # Configuration parameters
        self.p_value_threshold = getattr(self.config.pathway, 'p_value_threshold', 0.05)
        self.fdr_threshold = getattr(self.config.pathway, 'fdr_threshold', 0.1)
        self.min_pathway_size = getattr(self.config.pathway, 'min_pathway_size', 5)
        self.max_pathway_size = getattr(self.config.pathway, 'max_pathway_size', 500)
        
        logger.info(f"Initialized PathwayAnalyzer with databases: {databases}")
    
    def load_pathway_data(self, force_reload: bool = False) -> None:
        """
        Load pathway data from configured databases.
        
        Args:
            force_reload: Whether to reload data even if already cached
        """
        logger.info("Loading pathway data...")
        
        for database in self.databases:
            if database not in self.pathway_data or force_reload:
                logger.info(f"Loading {database} pathway data...")
                
                try:
                    if database == "KEGG":
                        self.pathway_data[database] = self._load_kegg_pathways()
                    elif database == "Reactome":
                        self.pathway_data[database] = self._load_reactome_pathways()
                    elif database == "GO":
                        self.pathway_data[database] = self._load_go_pathways()
                    elif database == "MSigDB":
                        self.pathway_data[database] = self._load_msigdb_pathways()
                    else:
                        logger.warning(f"Unknown database: {database}")
                        continue
                    
                    logger.info(f"Loaded {len(self.pathway_data[database])} pathways from {database}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {database} data: {str(e)}")
                    # Create placeholder data
                    self.pathway_data[database] = self._create_placeholder_pathways(database)
        
        logger.info("Pathway data loading completed")
    
    def _load_kegg_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Load KEGG pathway data (placeholder implementation)."""
        # In a real implementation, this would query the KEGG API
        # For now, create placeholder data
        return self._create_placeholder_pathways("KEGG")
    
    def _load_reactome_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Load Reactome pathway data (placeholder implementation)."""
        # In a real implementation, this would query the Reactome API
        return self._create_placeholder_pathways("Reactome")
    
    def _load_go_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Load Gene Ontology pathway data (placeholder implementation)."""
        # In a real implementation, this would query the GO API
        return self._create_placeholder_pathways("GO")
    
    def _load_msigdb_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Load MSigDB pathway data (placeholder implementation)."""
        # In a real implementation, this would load MSigDB gene sets
        return self._create_placeholder_pathways("MSigDB")
    
    def _create_placeholder_pathways(self, database: str) -> Dict[str, Dict[str, Any]]:
        """Create placeholder pathway data for testing."""
        placeholder_pathways = {}
        
        # Create some example pathways
        pathway_templates = [
            ("Cell Cycle", ["CCND1", "CDK4", "RB1", "E2F1", "CDKN1A"]),
            ("Apoptosis", ["TP53", "BAX", "BCL2", "CASP3", "CASP9"]),
            ("DNA Repair", ["BRCA1", "BRCA2", "ATM", "CHEK2", "RAD51"]),
            ("PI3K-AKT Signaling", ["PIK3CA", "AKT1", "PTEN", "mTOR", "GSK3B"]),
            ("MAPK Signaling", ["KRAS", "RAF1", "MEK1", "ERK1", "JUN"]),
            ("p53 Signaling", ["TP53", "MDM2", "CDKN1A", "BAX", "GADD45A"]),
            ("Wnt Signaling", ["WNT1", "CTNNB1", "APC", "GSK3B", "TCF7"]),
            ("TGF-beta Signaling", ["TGFB1", "SMAD2", "SMAD3", "SMAD4", "TGFBR1"]),
            ("Notch Signaling", ["NOTCH1", "DLL1", "HES1", "HEY1", "RBPJ"]),
            ("Hedgehog Signaling", ["SHH", "PTCH1", "SMO", "GLI1", "SUFU"])
        ]
        
        for i, (pathway_name, genes) in enumerate(pathway_templates):
            pathway_id = f"{database}_{i:03d}"
            full_pathway_name = f"{database}: {pathway_name}"
            
            placeholder_pathways[pathway_id] = {
                "name": full_pathway_name,
                "description": f"{pathway_name} pathway from {database}",
                "genes": genes,
                "size": len(genes),
                "database": database
            }
        
        return placeholder_pathways
    
    def run_analysis(self, 
                    gene_list: List[str], 
                    background_genes: Optional[List[str]] = None,
                    method: str = "ora",
                    **kwargs) -> pd.DataFrame:
        """
        Run pathway analysis on gene list.
        
        Args:
            gene_list: List of genes of interest
            background_genes: Background gene set (universe)
            method: Analysis method ("ora" for over-representation, "gsea" for gene set enrichment)
            **kwargs: Additional analysis parameters
            
        Returns:
            pd.DataFrame: Pathway analysis results
        """
        logger.info(f"Running pathway analysis using {method} method...")
        
        # Ensure pathway data is loaded
        if not self.pathway_data:
            self.load_pathway_data()
        
        # Validate input
        if not gene_list:
            raise ValueError("Gene list cannot be empty")
        
        gene_set = set(gene_list)
        
        if background_genes is None:
            # Use all genes from pathway databases as background
            background_genes = self._get_all_pathway_genes()
        
        background_set = set(background_genes)
        
        # Run analysis based on method
        if method == "ora":
            results = self._run_ora_analysis(gene_set, background_set, **kwargs)
        elif method == "gsea":
            # For GSEA, we need gene scores
            gene_scores = kwargs.get("gene_scores", {})
            results = self._run_gsea_analysis(gene_scores, **kwargs)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
        
        # Store results
        self.analysis_results[method] = results
        
        logger.info(f"Pathway analysis completed: {len(results)} pathways analyzed")
        
        return results
    
    def _run_ora_analysis(self, 
                         gene_set: Set[str], 
                         background_set: Set[str],
                         **kwargs) -> pd.DataFrame:
        """Run over-representation analysis."""
        results = []
        
        # Filter gene set to background
        gene_set = gene_set.intersection(background_set)
        n_query_genes = len(gene_set)
        n_background_genes = len(background_set)
        
        logger.info(f"ORA analysis: {n_query_genes} query genes, {n_background_genes} background genes")
        
        # Test each pathway
        for database in self.databases:
            if database not in self.pathway_data:
                continue
            
            for pathway_id, pathway_info in self.pathway_data[database].items():
                pathway_genes = set(pathway_info["genes"])
                
                # Filter pathway genes to background
                pathway_genes = pathway_genes.intersection(background_set)
                n_pathway_genes = len(pathway_genes)
                
                # Skip pathways outside size limits
                if n_pathway_genes < self.min_pathway_size or n_pathway_genes > self.max_pathway_size:
                    continue
                
                # Calculate overlap
                overlap_genes = gene_set.intersection(pathway_genes)
                n_overlap = len(overlap_genes)
                
                if n_overlap == 0:
                    continue
                
                # Hypergeometric test
                p_value = hypergeom.sf(
                    n_overlap - 1,  # Number of successes - 1 (for survival function)
                    n_background_genes,  # Population size
                    n_pathway_genes,  # Number of success states in population
                    n_query_genes  # Number of draws
                )
                
                # Calculate enrichment ratio
                expected = (n_query_genes * n_pathway_genes) / n_background_genes
                enrichment_ratio = n_overlap / expected if expected > 0 else 0
                
                results.append({
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_info["name"],
                    "database": pathway_info["database"],
                    "pathway_size": n_pathway_genes,
                    "overlap_size": n_overlap,
                    "query_size": n_query_genes,
                    "p_value": p_value,
                    "enrichment_ratio": enrichment_ratio,
                    "overlap_genes": list(overlap_genes)
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            logger.warning("No significant pathways found")
            return results_df
        
        # Multiple testing correction
        results_df["fdr"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
        
        # Sort by p-value
        results_df = results_df.sort_values("p_value")
        
        # Filter by significance thresholds
        significant_results = results_df[
            (results_df["p_value"] <= self.p_value_threshold) |
            (results_df["fdr"] <= self.fdr_threshold)
        ]
        
        logger.info(f"Found {len(significant_results)} significant pathways")
        
        return results_df
    
    def _run_gsea_analysis(self, gene_scores: Dict[str, float], **kwargs) -> pd.DataFrame:
        """Run gene set enrichment analysis (simplified implementation)."""
        # This is a simplified GSEA implementation
        # In practice, you would use libraries like gseapy or implement the full GSEA algorithm
        
        results = []
        
        if not gene_scores:
            logger.warning("No gene scores provided for GSEA")
            return pd.DataFrame()
        
        # Sort genes by score
        sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
        gene_ranks = {gene: rank for rank, (gene, score) in enumerate(sorted_genes)}
        
        # Test each pathway
        for database in self.databases:
            if database not in self.pathway_data:
                continue
            
            for pathway_id, pathway_info in self.pathway_data[database].items():
                pathway_genes = set(pathway_info["genes"])
                
                # Get genes in pathway that have scores
                pathway_genes_with_scores = pathway_genes.intersection(set(gene_scores.keys()))
                
                if len(pathway_genes_with_scores) < self.min_pathway_size:
                    continue
                
                # Calculate enrichment score (simplified)
                pathway_scores = [gene_scores[gene] for gene in pathway_genes_with_scores]
                mean_pathway_score = np.mean(pathway_scores)
                
                # Simple statistical test (in practice, use proper GSEA statistics)
                all_scores = list(gene_scores.values())
                t_stat, p_value = stats.ttest_1samp(pathway_scores, np.mean(all_scores))
                
                results.append({
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_info["name"],
                    "database": pathway_info["database"],
                    "pathway_size": len(pathway_genes_with_scores),
                    "enrichment_score": mean_pathway_score,
                    "p_value": p_value,
                    "pathway_genes": list(pathway_genes_with_scores)
                })
        
        # Convert to DataFrame and process
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            results_df["fdr"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
            results_df = results_df.sort_values("p_value")
        
        return results_df
    
    def _get_all_pathway_genes(self) -> List[str]:
        """Get all genes from all pathway databases."""
        all_genes = set()
        
        for database in self.pathway_data:
            for pathway_info in self.pathway_data[database].values():
                all_genes.update(pathway_info["genes"])
        
        return list(all_genes)
    
    def get_pathways(self, database: Optional[str] = None, **kwargs) -> Dict[str, List[str]]:
        """
        Get pathway gene sets.
        
        Args:
            database: Specific database to get pathways from
            **kwargs: Additional filtering parameters
            
        Returns:
            Dict[str, List[str]]: Pathway ID to gene list mapping
        """
        if not self.pathway_data:
            self.load_pathway_data()
        
        pathways = {}
        
        databases_to_use = [database] if database else self.databases
        
        for db in databases_to_use:
            if db in self.pathway_data:
                for pathway_id, pathway_info in self.pathway_data[db].items():
                    pathways[pathway_id] = pathway_info["genes"]
        
        return pathways
    
    def get_pathway_network(self, 
                           significant_pathways: Optional[List[str]] = None,
                           similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Build pathway similarity network.
        
        Args:
            significant_pathways: List of pathway IDs to include
            similarity_threshold: Minimum similarity for network edges
            
        Returns:
            Dict[str, Any]: Network data with nodes and edges
        """
        if not self.pathway_data:
            self.load_pathway_data()
        
        # Get pathways to include
        if significant_pathways is None:
            # Use all pathways
            pathways_to_include = {}
            for db in self.pathway_data:
                pathways_to_include.update(self.pathway_data[db])
        else:
            # Use only specified pathways
            pathways_to_include = {}
            for db in self.pathway_data:
                for pathway_id, pathway_info in self.pathway_data[db].items():
                    if pathway_id in significant_pathways:
                        pathways_to_include[pathway_id] = pathway_info
        
        # Calculate pathway similarities (Jaccard index)
        pathway_ids = list(pathways_to_include.keys())
        nodes = []
        edges = []
        
        for pathway_id in pathway_ids:
            pathway_info = pathways_to_include[pathway_id]
            nodes.append({
                "id": pathway_id,
                "name": pathway_info["name"],
                "size": pathway_info["size"],
                "database": pathway_info["database"]
            })
        
        # Calculate edges
        for i, pathway_id1 in enumerate(pathway_ids):
            genes1 = set(pathways_to_include[pathway_id1]["genes"])
            
            for j, pathway_id2 in enumerate(pathway_ids[i+1:], i+1):
                genes2 = set(pathways_to_include[pathway_id2]["genes"])
                
                # Jaccard similarity
                intersection = len(genes1.intersection(genes2))
                union = len(genes1.union(genes2))
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity >= similarity_threshold:
                        edges.append({
                            "source": pathway_id1,
                            "target": pathway_id2,
                            "weight": similarity,
                            "shared_genes": intersection
                        })
        
        network = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "n_pathways": len(nodes),
                "n_edges": len(edges),
                "similarity_threshold": similarity_threshold
            }
        }
        
        logger.info(f"Built pathway network: {len(nodes)} nodes, {len(edges)} edges")
        
        return network
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save analysis results to file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        
        # Save all analysis results
        for method, results in self.analysis_results.items():
            method_path = output_path.parent / f"{output_path.stem}_{method}{output_path.suffix}"
            results.to_csv(method_path, index=False)
            logger.info(f"Saved {method} results to {method_path}")


# Export main classes
__all__ = [
    "PathwayAnalyzer",
    "PathwayAnalysisBase"
]