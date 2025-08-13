#!/usr/bin/env python3
"""
Multi-omics Biomarker Discovery Demo
====================================

This demo script generates synthetic multi-omics data and runs a complete
biomarker discovery analysis pipeline including:

1. Synthetic data generation (mimicking GDSC-like data)
2. Multi-omics data preprocessing
3. Feature integration
4. Drug response prediction modeling
5. Biomarker discovery
6. Results visualization and reporting

This demonstrates the full workflow without requiring actual GDSC data download.
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_multiomics_data(n_samples=500, n_genes=2000, n_drugs=10, random_state=42):
    """Generate synthetic multi-omics data mimicking GDSC structure."""
    
    print("Generating synthetic multi-omics data...")
    np.random.seed(random_state)
    
    # Generate sample IDs
    sample_ids = [f"CELL_LINE_{i:04d}" for i in range(n_samples)]
    gene_ids = [f"GENE_{i:04d}" for i in range(n_genes)]
    drug_names = [f"Drug_{chr(65+i)}" for i in range(n_drugs)]
    
    # 1. Gene Expression Data (RNA-seq like)
    # Simulate log-normal distribution typical of gene expression
    expression_data = pd.DataFrame(
        np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes)),
        index=sample_ids,
        columns=gene_ids
    )
    
    # 2. Genomics Data (Mutations + CNV)
    # Binary mutation data (0/1)
    mutation_data = pd.DataFrame(
        np.random.binomial(1, 0.1, size=(n_samples, n_genes//4)),  # 10% mutation rate
        index=sample_ids,
        columns=[f"MUT_{gene}" for gene in gene_ids[:n_genes//4]]
    )
    
    # Copy number variation data (-2 to +2)
    cnv_data = pd.DataFrame(
        np.random.normal(0, 0.5, size=(n_samples, n_genes//4)),
        index=sample_ids,
        columns=[f"CNV_{gene}" for gene in gene_ids[:n_genes//4]]
    )
    
    # 3. Drug Response Data
    # Simulate IC50 values (log scale)
    drug_response = pd.DataFrame(
        np.random.normal(1, 0.8, size=(n_samples, n_drugs)),
        index=sample_ids,
        columns=drug_names
    )
    
    # Add some realistic correlations between genomics and drug response
    # Make some genes predictive of drug response
    for i, drug in enumerate(drug_names):
        # Select some genes to be predictive
        predictive_genes = np.random.choice(n_genes, size=20, replace=False)
        for gene_idx in predictive_genes:
            gene_name = gene_ids[gene_idx]
            correlation_strength = np.random.uniform(0.3, 0.7)
            noise = np.random.normal(0, 0.2, n_samples)
            
            # Add correlation with expression
            drug_response[drug] += correlation_strength * (expression_data[gene_name] / expression_data[gene_name].std()) + noise
    
    # 4. Cell Line Metadata
    tissue_types = ['Lung', 'Breast', 'Colon', 'Brain', 'Skin', 'Blood', 'Liver', 'Kidney']
    cell_line_info = pd.DataFrame({
        'COSMIC_ID': sample_ids,
        'Cell_Line_Name': [f"CL_{i}" for i in range(n_samples)],
        'Tissue': np.random.choice(tissue_types, n_samples),
        'Cancer_Type': np.random.choice(['Carcinoma', 'Sarcoma', 'Leukemia', 'Lymphoma'], n_samples),
        'Growth_Rate': np.random.uniform(0.5, 2.0, n_samples)
    })
    cell_line_info.set_index('COSMIC_ID', inplace=True)
    
    print(f"✓ Generated synthetic data:")
    print(f"  - Expression: {expression_data.shape}")
    print(f"  - Mutations: {mutation_data.shape}")
    print(f"  - CNV: {cnv_data.shape}")
    print(f"  - Drug Response: {drug_response.shape}")
    print(f"  - Cell Line Info: {cell_line_info.shape}")
    
    return {
        'expression': expression_data,
        'mutations': mutation_data,
        'cnv': cnv_data,
        'drug_response': drug_response,
        'cell_lines': cell_line_info
    }

def preprocess_multiomics_data(data_dict):
    """Preprocess and integrate multi-omics data."""
    
    print("\nPreprocessing multi-omics data...")
    
    # 1. Preprocess expression data (log2 transform and standardize)
    expression = data_dict['expression']
    expression_log = np.log2(expression + 1)  # Add pseudocount
    expression_scaled = pd.DataFrame(
        StandardScaler().fit_transform(expression_log),
        index=expression_log.index,
        columns=expression_log.columns
    )
    
    # 2. Preprocess genomics data (already in good format)
    mutations = data_dict['mutations']
    cnv = data_dict['cnv']
    cnv_scaled = pd.DataFrame(
        StandardScaler().fit_transform(cnv),
        index=cnv.index,
        columns=cnv.columns
    )
    
    # 3. Integrate features (concatenate)
    # Select top variable genes for expression
    expression_var = expression_scaled.var().sort_values(ascending=False)
    top_genes = expression_var.head(1000).index  # Top 1000 most variable genes
    
    integrated_features = pd.concat([
        expression_scaled[top_genes].add_prefix('EXP_'),
        mutations.add_prefix('MUT_'),
        cnv_scaled.add_prefix('CNV_')
    ], axis=1)
    
    print(f"✓ Preprocessed data:")
    print(f"  - Integrated features: {integrated_features.shape}")
    
    return integrated_features, data_dict['drug_response']

def train_drug_response_models(X, y, target_drug):
    """Train multiple models for drug response prediction."""
    
    print(f"\nTraining models for {target_drug}...")
    
    # Get target drug response
    drug_response = y[target_drug].dropna()
    
    # Align features with available drug response data
    common_samples = X.index.intersection(drug_response.index)
    X_aligned = X.loc[common_samples]
    y_aligned = drug_response.loc[common_samples]
    
    print(f"✓ Aligned data: {len(common_samples)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )
    
    # Train multiple models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred_test
            }
        }
        
        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test R²: {test_r2:.4f}")
        print(f"    Test RMSE: {test_rmse:.4f}")
        print(f"    CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]
    
    print(f"✓ Best model: {best_model_name} (Test R² = {best_model['test_r2']:.4f})")
    
    return results, best_model_name

def discover_biomarkers(X, y, target_drug, n_features=50):
    """Discover biomarkers using feature selection."""
    
    print(f"\nDiscovering biomarkers for {target_drug}...")
    
    # Get target drug response
    drug_response = y[target_drug].dropna()
    
    # Align features
    common_samples = X.index.intersection(drug_response.index)
    X_aligned = X.loc[common_samples]
    y_aligned = drug_response.loc[common_samples]
    
    # Feature selection using univariate statistics
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_aligned, y_aligned)
    
    # Get selected feature names and scores
    selected_features = X_aligned.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    # Create biomarker dataframe
    biomarkers = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores,
        'feature_type': [feat.split('_')[0] for feat in selected_features]
    }).sort_values('score', ascending=False)
    
    print(f"✓ Discovered {len(biomarkers)} biomarkers")
    print("✓ Top 10 biomarkers:")
    for idx, row in biomarkers.head(10).iterrows():
        print(f"  {row['feature']}: {row['score']:.2f} ({row['feature_type']})")
    
    return biomarkers

def create_visualizations(results, biomarkers, target_drug, output_dir):
    """Create visualization plots."""
    
    print(f"\nCreating visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(results.keys())
    test_r2_scores = [results[name]['test_r2'] for name in model_names]
    cv_r2_means = [results[name]['cv_r2_mean'] for name in model_names]
    cv_r2_stds = [results[name]['cv_r2_std'] for name in model_names]
    
    # Test R² comparison
    bars1 = ax1.bar(model_names, test_r2_scores, color=['skyblue', 'lightcoral'])
    ax1.set_title('Model Performance (Test R²)')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, max(test_r2_scores) * 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars1, test_r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Cross-validation R² with error bars
    bars2 = ax2.bar(model_names, cv_r2_means, yerr=cv_r2_stds, 
                    capsize=5, color=['lightgreen', 'orange'])
    ax2.set_title('Cross-Validation Performance')
    ax2.set_ylabel('CV R² Score')
    ax2.set_ylim(0, max(cv_r2_means) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual scatter plot (best model)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_model_name]
    
    plt.figure(figsize=(8, 6))
    y_test = best_result['predictions']['y_test']
    y_pred = best_result['predictions']['y_pred']
    
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
    
    # Add diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.xlabel('Actual Drug Response')
    plt.ylabel('Predicted Drug Response')
    plt.title(f'{best_model_name} - Predictions vs Actual\n{target_drug} (R² = {best_result["test_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Biomarkers
    plt.figure(figsize=(10, 8))
    top_biomarkers = biomarkers.head(20)
    
    # Create color map for feature types
    feature_types = top_biomarkers['feature_type'].unique()
    colors = sns.color_palette("Set2", len(feature_types))
    color_map = dict(zip(feature_types, colors))
    bar_colors = [color_map[ft] for ft in top_biomarkers['feature_type']]
    
    bars = plt.barh(range(len(top_biomarkers)), top_biomarkers['score'], color=bar_colors)
    plt.yticks(range(len(top_biomarkers)), top_biomarkers['feature'])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top 20 Biomarkers for {target_drug}')
    plt.gca().invert_yaxis()
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[ft], label=ft) 
                      for ft in feature_types]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_biomarkers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Biomarker Type Distribution
    plt.figure(figsize=(8, 6))
    type_counts = biomarkers['feature_type'].value_counts()
    
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set3", len(type_counts)))
    plt.title('Distribution of Biomarker Types')
    
    plt.savefig(output_dir / 'biomarker_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualizations to {output_dir}")

def generate_report(results, biomarkers, target_drug, output_dir):
    """Generate analysis report."""
    
    print(f"\nGenerating analysis report...")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_model_name]
    
    report = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'target_drug': target_drug,
            'best_model': best_model_name
        },
        'model_performance': {
            name: {
                'test_r2': float(result['test_r2']),
                'test_rmse': float(result['test_rmse']),
                'cv_r2_mean': float(result['cv_r2_mean']),
                'cv_r2_std': float(result['cv_r2_std'])
            }
            for name, result in results.items()
        },
        'biomarkers': {
            'total_discovered': len(biomarkers),
            'by_type': biomarkers['feature_type'].value_counts().to_dict(),
            'top_10': biomarkers.head(10)[['feature', 'score', 'feature_type']].to_dict('records')
        }
    }
    
    # Save JSON report
    with open(Path(output_dir) / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save detailed biomarkers CSV
    biomarkers.to_csv(Path(output_dir) / 'discovered_biomarkers.csv', index=False)
    
    print(f"✓ Saved analysis report to {output_dir}")
    
    return report

def main():
    """Run the complete multi-omics biomarker discovery demo."""
    
    print("="*80)
    print("Multi-omics Biomarker Discovery Demo")
    print("="*80)
    
    output_dir = Path("results/demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Generate synthetic data
        data_dict = generate_synthetic_multiomics_data(
            n_samples=500, n_genes=2000, n_drugs=5, random_state=42
        )
        
        # Step 2: Preprocess and integrate data
        integrated_features, drug_response = preprocess_multiomics_data(data_dict)
        
        # Step 3: Select target drug (one with most complete data)
        drug_coverage = drug_response.count()
        target_drug = drug_coverage.idxmax()
        print(f"\n✓ Selected target drug: {target_drug} ({drug_coverage[target_drug]} samples)")
        
        # Step 4: Train prediction models
        results, best_model_name = train_drug_response_models(
            integrated_features, drug_response, target_drug
        )
        
        # Step 5: Discover biomarkers
        biomarkers = discover_biomarkers(
            integrated_features, drug_response, target_drug, n_features=100
        )
        
        # Step 6: Create visualizations
        create_visualizations(results, biomarkers, target_drug, output_dir)
        
        # Step 7: Generate report
        report = generate_report(results, biomarkers, target_drug, output_dir)
        
        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Target Drug: {target_drug}")
        print(f"Best Model: {best_model_name}")
        print(f"Best Test R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"Biomarkers Discovered: {len(biomarkers)}")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Multi-omics biomarker discovery demo completed successfully!")
        print("Check the results/demo_output directory for detailed outputs and visualizations.")
    else:
        print("\n💥 Demo failed. Check the logs for details.")
        sys.exit(1)