"""
Extended pathway configuration for Multi-omics Biomarker Discovery

This module provides extended configuration classes for pathway analysis
and network analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class PathwayConfig:
    """Extended configuration for pathway analysis"""
    
    # Database Selection
    databases: List[str] = field(default_factory=lambda: [
        "KEGG", "Reactome", "GO", "MSigDB"
    ])
    
    # Statistical Thresholds
    p_value_threshold: float = 0.05
    fdr_threshold: float = 0.1
    min_pathway_size: int = 5
    max_pathway_size: int = 500
    
    # Gene Set Enrichment Analysis
    gsea_permutations: int = 1000
    gsea_scoring_class: str = "weighted"  # weighted, classic
    gsea_metric: str = "signal_to_noise"
    
    # Network Analysis
    network_edge_threshold: float = 0.5
    network_min_degree: int = 2
    network_layout: str = "spring"  # spring, circular, kamada_kawai
    
    # GO Analysis Configuration
    go_aspects: List[str] = field(default_factory=lambda: [
        "biological_process", "molecular_function", "cellular_component"
    ])
    go_evidence_codes: List[str] = field(default_factory=lambda: [
        "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"
    ])
    
    # KEGG Configuration
    kegg_organisms: List[str] = field(default_factory=lambda: ["hsa"])  # homo sapiens
    kegg_pathway_classes: List[str] = field(default_factory=lambda: [
        "Metabolism", "Genetic Information Processing", 
        "Environmental Information Processing", "Cellular Processes",
        "Organismal Systems", "Human Diseases"
    ])
    
    # Reactome Configuration
    reactome_species: str = "Homo sapiens"
    reactome_include_interactors: bool = True
    
    # MSigDB Configuration
    msigdb_collections: Dict[str, str] = field(default_factory=lambda: {
        "H": "Hallmark gene sets",
        "C1": "Positional gene sets", 
        "C2": "Curated gene sets",
        "C3": "Regulatory target gene sets",
        "C4": "Computational gene sets",
        "C5": "Ontology gene sets",
        "C6": "Oncogenic signature gene sets",
        "C7": "Immunologic signature gene sets",
        "C8": "Cell type signature gene sets"
    })
    
    def get_enrichment_methods(self) -> List[str]:
        """Get available enrichment analysis methods"""
        return [
            "fisher_exact", "hypergeometric", "gsea", 
            "ssgsea", "gsva", "plage"
        ]
    
    def get_network_metrics(self) -> List[str]:
        """Get network analysis metrics to compute"""
        return [
            "degree_centrality", "betweenness_centrality", 
            "closeness_centrality", "eigenvector_centrality",
            "clustering_coefficient", "pagerank"
        ]
    
    def get_visualization_options(self) -> Dict[str, Any]:
        """Get pathway visualization options"""
        return {
            "node_size_range": (20, 200),
            "edge_width_range": (0.5, 5.0),
            "color_schemes": ["viridis", "plasma", "inferno", "magma"],
            "layout_algorithms": ["spring", "circular", "kamada_kawai", "shell"],
            "node_shapes": ["circle", "square", "triangle", "diamond"]
        }
    
    def get_pathway_databases_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about pathway databases"""
        return {
            "KEGG": {
                "full_name": "Kyoto Encyclopedia of Genes and Genomes",
                "url": "https://www.genome.jp/kegg/",
                "data_types": ["pathways", "modules", "diseases", "drugs"]
            },
            "Reactome": {
                "full_name": "Reactome Pathway Database",
                "url": "https://reactome.org/",
                "data_types": ["pathways", "reactions", "complexes", "interactions"]
            },
            "GO": {
                "full_name": "Gene Ontology",
                "url": "http://geneontology.org/",
                "data_types": ["biological_process", "molecular_function", "cellular_component"]
            },
            "MSigDB": {
                "full_name": "Molecular Signatures Database",
                "url": "https://www.gsea-msigdb.org/gsea/msigdb/",
                "data_types": ["gene_sets", "signatures", "collections"]
            }
        }


__all__ = ["PathwayConfig"]