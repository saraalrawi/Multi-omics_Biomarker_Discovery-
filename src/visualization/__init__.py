"""
Visualization Module for Multi-omics Biomarker Discovery

This module provides comprehensive visualization capabilities including:
- Multi-omics data visualization
- Model performance plots
- Biomarker analysis visualizations
- Pathway network visualizations
- Interactive plots and dashboards
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Optional advanced visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..utils import setup_logging, load_config, ensure_directory


logger = logging.getLogger(__name__)

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def setup_matplotlib_style():
    """Set up consistent matplotlib styling."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_data_overview(data: Dict[str, pd.DataFrame], 
                      output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create overview plots of multi-omics data.
    
    Args:
        data: Dictionary of DataFrames with omics data
        output_path: Optional path to save the plot
        
    Returns:
        plt.Figure: The created figure
    """
    setup_matplotlib_style()
    
    n_omics = len(data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-omics Data Overview', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot 1: Data shapes
    omics_types = list(data.keys())
    n_samples = [df.shape[0] for df in data.values()]
    n_features = [df.shape[1] for df in data.values()]
    
    x_pos = np.arange(len(omics_types))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, n_samples, width, label='Samples', alpha=0.8)
    axes[0].bar(x_pos + width/2, n_features, width, label='Features', alpha=0.8)
    axes[0].set_xlabel('Omics Type')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Data Dimensions by Omics Type')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(omics_types, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Missing data heatmap
    missing_data = {}
    for omics_type, df in data.items():
        missing_rates = df.isnull().sum() / len(df)
        missing_data[omics_type] = missing_rates.head(20)  # Top 20 features
    
    if missing_data:
        # Create combined missing data DataFrame
        max_features = max(len(v) for v in missing_data.values())
        missing_df = pd.DataFrame(index=range(max_features))
        
        for omics_type, missing_rates in missing_data.items():
            missing_df[omics_type] = missing_rates.values[:max_features]
        
        missing_df = missing_df.fillna(0)
        
        sns.heatmap(missing_df.T, ax=axes[1], cmap='Reds', 
                   cbar_kws={'label': 'Missing Rate'})
        axes[1].set_title('Missing Data Patterns (Top 20 Features)')
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('Omics Type')
    
    # Plot 3: Data distribution example
    if data:
        first_omics = list(data.keys())[0]
        first_df = data[first_omics]
        
        # Plot distribution of first numeric column
        numeric_cols = first_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sample_col = numeric_cols[0]
            sample_data = first_df[sample_col].dropna()
            
            axes[2].hist(sample_data, bins=30, alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('Value')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'Distribution Example: {sample_col}')
            axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Sample overlap
    if len(data) > 1:
        sample_sets = {omics_type: set(df.index) for omics_type, df in data.items()}
        
        # Calculate pairwise overlaps
        overlap_matrix = pd.DataFrame(index=omics_types, columns=omics_types)
        
        for i, omics1 in enumerate(omics_types):
            for j, omics2 in enumerate(omics_types):
                if i == j:
                    overlap_matrix.loc[omics1, omics2] = len(sample_sets[omics1])
                else:
                    overlap = len(sample_sets[omics1].intersection(sample_sets[omics2]))
                    overlap_matrix.loc[omics1, omics2] = overlap
        
        overlap_matrix = overlap_matrix.astype(float)
        
        sns.heatmap(overlap_matrix, ax=axes[3], annot=True, fmt='.0f', 
                   cmap='Blues', cbar_kws={'label': 'Sample Overlap'})
        axes[3].set_title('Sample Overlap Between Omics Types')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Data overview plot saved to {output_path}")
    
    return fig


def plot_model_performance(performance_results: Dict[str, Dict[str, float]], 
                          output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot model performance comparison.
    
    Args:
        performance_results: Dictionary of model performance metrics
        output_path: Optional path to save the plot
        
    Returns:
        plt.Figure: The created figure
    """
    setup_matplotlib_style()
    
    if not performance_results:
        raise ValueError("No performance results provided")
    
    # Convert to DataFrame for easier plotting
    perf_df = pd.DataFrame(performance_results).T
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Plot 1: R² scores
    if 'r2_score' in perf_df.columns:
        perf_df['r2_score'].plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8)
        axes[0].set_title('R² Score by Model')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: RMSE
    if 'rmse' in perf_df.columns:
        perf_df['rmse'].plot(kind='bar', ax=axes[1], color='lightcoral', alpha=0.8)
        axes[1].set_title('RMSE by Model')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: MAE
    if 'mae' in perf_df.columns:
        perf_df['mae'].plot(kind='bar', ax=axes[2], color='lightgreen', alpha=0.8)
        axes[2].set_title('MAE by Model')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison radar chart (simplified)
    if len(perf_df.columns) >= 2:
        # Normalize metrics for comparison (0-1 scale)
        normalized_perf = perf_df.copy()
        
        # For R², higher is better (keep as is, assuming 0-1 range)
        if 'r2_score' in normalized_perf.columns:
            normalized_perf['r2_score'] = np.clip(normalized_perf['r2_score'], 0, 1)
        
        # For error metrics, lower is better (invert)
        error_metrics = ['rmse', 'mae', 'mse']
        for metric in error_metrics:
            if metric in normalized_perf.columns:
                max_val = normalized_perf[metric].max()
                if max_val > 0:
                    normalized_perf[metric] = 1 - (normalized_perf[metric] / max_val)
        
        # Plot normalized metrics
        normalized_perf.plot(kind='bar', ax=axes[3], alpha=0.8)
        axes[3].set_title('Normalized Performance Metrics')
        axes[3].set_ylabel('Normalized Score (Higher = Better)')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model performance plot saved to {output_path}")
    
    return fig


def plot_biomarker_analysis(biomarkers: pd.DataFrame, 
                           top_n: int = 20,
                           output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot biomarker analysis results.
    
    Args:
        biomarkers: DataFrame with biomarker information
        top_n: Number of top biomarkers to display
        output_path: Optional path to save the plot
        
    Returns:
        plt.Figure: The created figure
    """
    setup_matplotlib_style()
    
    if biomarkers.empty:
        raise ValueError("No biomarkers provided")
    
    # Take top N biomarkers
    top_biomarkers = biomarkers.head(top_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Biomarkers Analysis', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Plot 1: Biomarker scores
    score_col = None
    for col in ['stability_score', 'ensemble_score', 'correlation', 'abs_coefficient']:
        if col in top_biomarkers.columns:
            score_col = col
            break
    
    if score_col:
        y_pos = np.arange(len(top_biomarkers))
        axes[0].barh(y_pos, top_biomarkers[score_col], alpha=0.8)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(top_biomarkers['feature'], fontsize=8)
        axes[0].set_xlabel(score_col.replace('_', ' ').title())
        axes[0].set_title(f'Biomarker {score_col.replace("_", " ").title()}')
        axes[0].grid(True, alpha=0.3)
        axes[0].invert_yaxis()
    
    # Plot 2: Omics type distribution
    if 'omics_type' in top_biomarkers.columns:
        omics_counts = top_biomarkers['omics_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(omics_counts)))
        
        wedges, texts, autotexts = axes[1].pie(omics_counts.values, 
                                              labels=omics_counts.index,
                                              autopct='%1.1f%%',
                                              colors=colors)
        axes[1].set_title('Biomarkers by Omics Type')
    
    # Plot 3: Correlation with target (if available)
    if 'target_correlation' in top_biomarkers.columns:
        correlations = top_biomarkers['target_correlation'].dropna()
        
        if not correlations.empty:
            axes[2].scatter(range(len(correlations)), correlations, alpha=0.7, s=50)
            axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[2].set_xlabel('Biomarker Rank')
            axes[2].set_ylabel('Correlation with Target')
            axes[2].set_title('Target Correlation by Biomarker')
            axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Missing data rates (if available)
    if 'missing_rate' in top_biomarkers.columns:
        missing_rates = top_biomarkers['missing_rate'].dropna()
        
        if not missing_rates.empty:
            axes[3].hist(missing_rates, bins=15, alpha=0.7, edgecolor='black')
            axes[3].set_xlabel('Missing Data Rate')
            axes[3].set_ylabel('Number of Biomarkers')
            axes[3].set_title('Distribution of Missing Data Rates')
            axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Biomarker analysis plot saved to {output_path}")
    
    return fig


def plot_pathway_network(network_data: Dict[str, Any], 
                        output_path: Optional[Union[str, Path]] = None,
                        layout: str = "spring") -> plt.Figure:
    """
    Plot pathway similarity network.
    
    Args:
        network_data: Network data with nodes and edges
        output_path: Optional path to save the plot
        layout: Network layout algorithm
        
    Returns:
        plt.Figure: The created figure
    """
    if not HAS_NETWORKX:
        logger.warning("NetworkX not available, cannot create network plot")
        return None
    
    setup_matplotlib_style()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for node in network_data['nodes']:
        G.add_node(node['id'], **node)
    
    # Add edges
    for edge in network_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw network
    # Node colors by database
    node_colors = []
    database_colors = {'KEGG': 'lightblue', 'Reactome': 'lightgreen', 
                      'GO': 'lightcoral', 'MSigDB': 'lightyellow'}
    
    for node_id in G.nodes():
        database = G.nodes[node_id].get('database', 'unknown')
        node_colors.append(database_colors.get(database, 'lightgray'))
    
    # Node sizes by pathway size
    node_sizes = []
    for node_id in G.nodes():
        size = G.nodes[node_id].get('size', 10)
        node_sizes.append(max(100, min(1000, size * 10)))  # Scale between 100-1000
    
    # Edge weights
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]  # Scale for visibility
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, ax=ax)
    
    # Add labels for important nodes (optional, can be crowded)
    if len(G.nodes()) <= 20:
        labels = {node_id: G.nodes[node_id].get('name', node_id)[:20] + '...' 
                 if len(G.nodes[node_id].get('name', node_id)) > 20 
                 else G.nodes[node_id].get('name', node_id)
                 for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Create legend
    legend_elements = [mpatches.Patch(color=color, label=database) 
                      for database, color in database_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f'Pathway Similarity Network\n({len(G.nodes())} pathways, {len(G.edges())} connections)')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Pathway network plot saved to {output_path}")
    
    return fig


def plot_feature_importance(importance_data: pd.DataFrame, 
                           top_n: int = 20,
                           output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        importance_data: DataFrame with feature importance scores
        top_n: Number of top features to display
        output_path: Optional path to save the plot
        
    Returns:
        plt.Figure: The created figure
    """
    setup_matplotlib_style()
    
    if importance_data.empty:
        raise ValueError("No importance data provided")
    
    # Take top N features
    top_features = importance_data.head(top_n)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    importance_col = 'importance' if 'importance' in top_features.columns else top_features.columns[1]
    
    bars = ax.barh(y_pos, top_features[importance_col], alpha=0.8)
    
    # Color bars by omics type if available
    if 'omics_type' in top_features.columns:
        omics_types = top_features['omics_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(omics_types)))
        omics_color_map = dict(zip(omics_types, colors))
        
        for i, (_, row) in enumerate(top_features.iterrows()):
            omics_type = row.get('omics_type', 'unknown')
            bars[i].set_color(omics_color_map.get(omics_type, 'gray'))
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=omics_type) 
                          for omics_type, color in omics_color_map.items()]
        ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel(importance_col.replace('_', ' ').title())
    ax.set_title(f'Top {top_n} Feature Importance Scores')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    return fig


def create_interactive_dashboard(data: Dict[str, Any], 
                               output_path: Optional[Union[str, Path]] = None) -> Optional[Any]:
    """
    Create interactive dashboard using Plotly (if available).
    
    Args:
        data: Dictionary containing various analysis results
        output_path: Optional path to save the HTML dashboard
        
    Returns:
        Plotly figure object or None if Plotly not available
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available, cannot create interactive dashboard")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Overview', 'Model Performance', 
                       'Top Biomarkers', 'Pathway Analysis'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Add plots based on available data
    if 'model_performance' in data:
        perf_data = data['model_performance']
        models = list(perf_data.keys())
        r2_scores = [perf_data[model].get('r2_score', 0) for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score'),
            row=1, col=2
        )
    
    if 'biomarkers' in data:
        biomarkers = data['biomarkers'].head(10)
        score_col = 'stability_score' if 'stability_score' in biomarkers.columns else biomarkers.columns[1]
        
        fig.add_trace(
            go.Bar(x=biomarkers['feature'], y=biomarkers[score_col], name='Biomarker Score'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="Multi-omics Analysis Dashboard",
        showlegend=False,
        height=800
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Interactive dashboard saved to {output_path}")
    
    return fig


# Export main functions
__all__ = [
    "setup_matplotlib_style",
    "plot_data_overview",
    "plot_model_performance", 
    "plot_biomarker_analysis",
    "plot_pathway_network",
    "plot_feature_importance",
    "create_interactive_dashboard"
]