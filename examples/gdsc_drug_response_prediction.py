#!/usr/bin/env python3
"""
GDSC Multi-omics Drug Response Prediction Example

This example demonstrates the complete workflow for multi-omics biomarker discovery
and drug response prediction using GDSC data, including:

1. Data acquisition from GDSC database
2. Multi-omics data preprocessing (genomics, transcriptomics, drug sensitivity)
3. Data integration and feature selection
4. Drug response prediction modeling
5. Biomarker discovery and validation
6. Pathway enrichment analysis
7. Results visualization

This example showcases the integration of genomics, transcriptomics, and drug
sensitivity data for comprehensive drug response prediction.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import from installed package
from data_acquisition.gdsc import GDSCDataAcquisition
from preprocessing.multiomics import MultiOmicsPreprocessor
from integration import MultiOmicsIntegrator
from modeling import DrugResponsePredictor, DrugResponseEvaluator, MultiDrugResponsePredictor
from biomarker_discovery import BiomarkerDiscovery
from pathway_analysis import PathwayAnalyzer
from visualization import (
    plot_data_overview, plot_model_performance,
    plot_biomarker_analysis, plot_pathway_network
)
from utils import setup_logging, load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating GDSC multi-omics drug response prediction."""
    
    print("="*80)
    print("GDSC Multi-omics Drug Response Prediction Example")
    print("="*80)
    
    try:
        # Configuration
        output_dir = Path("results/gdsc_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Data Acquisition
        print("\n1. GDSC Data Acquisition")
        print("-" * 40)
        
        gdsc_data = GDSCDataAcquisition()
        
        # Download multi-omics data
        success = gdsc_data.download(
            data_types=["drug_sensitivity", "genomics", "transcriptomics", "cell_lines"],
            force_download=False
        )
        
        if not success:
            raise Exception("Failed to download GDSC data")
        
        # Validate downloaded data
        validation_success = gdsc_data.validate()
        if not validation_success:
            raise Exception("GDSC data validation failed")
        
        print("✓ GDSC data acquisition completed successfully")
        
        # Get data summary
        data_summary = gdsc_data.get_data_summary()
        print(f"✓ Data summary: {data_summary}")
        
        # Step 2: Load and Preprocess Data
        print("\n2. Multi-omics Data Preprocessing")
        print("-" * 40)
        
        # Load raw data
        data_dir = Path("data/gdsc_cache")
        
        drug_sensitivity = pd.read_csv(data_dir / "drug_sensitivity_processed.csv")
        mutations = pd.read_csv(data_dir / "mutations_processed.csv")
        cnv = pd.read_csv(data_dir / "cnv_processed.csv")
        expression = pd.read_csv(data_dir / "expression_processed.csv")
        cell_lines = pd.read_csv(data_dir / "cell_lines_processed.csv")
        
        print(f"✓ Loaded data:")
        print(f"  - Drug sensitivity: {drug_sensitivity.shape}")
        print(f"  - Mutations: {mutations.shape}")
        print(f"  - CNV: {cnv.shape}")
        print(f"  - Expression: {expression.shape}")
        print(f"  - Cell lines: {cell_lines.shape}")
        
        # Initialize preprocessor
        preprocessor = MultiOmicsPreprocessor()
        
        # Preprocess drug response data
        drug_response_matrix, filtered_drug_data = preprocessor.preprocess_drug_response_data(
            drug_sensitivity, target_metric='LN_IC50'
        )
        
        # Preprocess genomics data
        genomics_features = preprocessor.preprocess_gdsc_genomics(mutations, cnv)
        
        # Preprocess expression data
        expression_features = preprocessor.preprocess_gdsc_expression(expression)
        
        print(f"✓ Preprocessed data:")
        print(f"  - Drug response matrix: {drug_response_matrix.shape}")
        print(f"  - Genomics features: {genomics_features.shape}")
        print(f"  - Expression features: {expression_features.shape}")
        
        # Step 3: Select Target Drug and Integrate Data
        print("\n3. Data Integration for Drug Response Prediction")
        print("-" * 40)
        
        # Select a drug with good coverage for demonstration
        drug_coverage = drug_response_matrix.count()
        target_drug = drug_coverage.idxmax()  # Drug with most data points
        
        print(f"✓ Selected target drug: {target_drug} ({drug_coverage[target_drug]} cell lines)")
        
        # Integrate multi-omics data for the target drug
        integrated_features, target_response = preprocessor.integrate_multiomics_for_drug_response(
            drug_response_matrix, genomics_features, expression_features, target_drug
        )
        
        print(f"✓ Integrated dataset: {integrated_features.shape[0]} samples × {integrated_features.shape[1]} features")
        
        # Step 4: Drug Response Prediction Modeling
        print("\n4. Drug Response Prediction Modeling")
        print("-" * 40)
        
        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            integrated_features, target_response, 
            test_size=0.2, random_state=42
        )
        
        print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")
        
        # Train drug response predictor
        predictor = DrugResponsePredictor(
            algorithms=["ridge", "random_forest", "xgboost"]
        )
        
        predictor.fit(X_train, y_train, optimize_hyperparameters=True)
        
        # Evaluate model
        evaluator = DrugResponseEvaluator()
        test_metrics = evaluator.evaluate_drug_response_model(
            predictor.best_model['model'], X_test, y_test, target_drug
        )
        
        print(f"✓ Best model: {predictor.best_model['algorithm']}")
        print(f"✓ Test performance: R² = {test_metrics['r2_score']:.4f}, RMSE = {test_metrics['rmse']:.4f}")
        print(f"✓ Pearson correlation: {test_metrics['pearson_correlation']:.4f}")
        
        # Cross-validation
        cv_results = evaluator.cross_validate_drug_response(
            predictor, integrated_features, target_response, target_drug, cv_folds=5
        )
        
        print(f"✓ Cross-validation R²: {cv_results['mean_metrics']['r2_score']:.4f} ± {cv_results['std_metrics']['r2_score']:.4f}")
        
        # Step 5: Biomarker Discovery
        print("\n5. Biomarker Discovery")
        print("-" * 40)
        
        biomarker_discovery = BiomarkerDiscovery(feature_selection_method="stability_selection")
        
        # Discover biomarkers
        biomarkers = biomarker_discovery.discover_biomarkers(
            integrated_features, target_response
        )
        
        print(f"✓ Discovered {len(biomarkers)} biomarkers")
        
        if len(biomarkers) > 0:
            print("✓ Top 10 biomarkers:")
            top_biomarkers = biomarkers.head(10)
            for idx, biomarker in top_biomarkers.iterrows():
                print(f"  - {biomarker['feature']}: {biomarker.get('stability_score', 'N/A'):.4f}")
            
            # Validate biomarkers
            validation_results = biomarker_discovery.validate_biomarkers(
                biomarkers['feature'].head(20).tolist(),
                integrated_features, target_response
            )
            
            print(f"✓ Biomarker validation CV score: {validation_results['mean_cv_score']:.4f}")
        
        # Step 6: Pathway Analysis
        print("\n6. Pathway Enrichment Analysis")
        print("-" * 40)
        
        if len(biomarkers) > 0:
            # Extract gene names from biomarker features
            biomarker_genes = []
            for feature in biomarkers['feature'].head(50):  # Top 50 biomarkers
                if 'expression_' in feature:
                    gene = feature.replace('expression_', '')
                    biomarker_genes.append(gene)
                elif 'genomics_' in feature:
                    gene = feature.replace('genomics_', '')
                    biomarker_genes.append(gene)
            
            if biomarker_genes:
                pathway_analyzer = PathwayAnalyzer()
                pathway_analyzer.load_pathway_data()
                
                # Run pathway analysis
                pathway_results = pathway_analyzer.run_analysis(
                    biomarker_genes, method="ora"
                )
                
                print(f"✓ Pathway analysis completed: {len(pathway_results)} pathways analyzed")
                
                if len(pathway_results) > 0:
                    significant_pathways = pathway_results[pathway_results['p_value'] < 0.05]
                    print(f"✓ Significant pathways: {len(significant_pathways)}")
                    
                    if len(significant_pathways) > 0:
                        print("✓ Top significant pathways:")
                        for idx, pathway in significant_pathways.head(5).iterrows():
                            print(f"  - {pathway['pathway_name']}: p = {pathway['p_value']:.4e}")
        
        # Step 7: Multi-drug Analysis (Optional)
        print("\n7. Multi-drug Response Prediction")
        print("-" * 40)
        
        # Select top drugs with good coverage for multi-drug analysis
        top_drugs = drug_coverage.nlargest(5).index.tolist()
        print(f"✓ Analyzing top {len(top_drugs)} drugs: {top_drugs}")
        
        multi_predictor = MultiDrugResponsePredictor(
            algorithms=["ridge", "random_forest"]
        )
        
        # Fit models for multiple drugs
        multi_results = multi_predictor.fit_multiple_drugs(
            integrated_features, drug_response_matrix, drugs_to_model=top_drugs
        )
        
        print(f"✓ Multi-drug modeling: {len(multi_results['successful_drugs'])}/{len(top_drugs)} successful")
        
        # Generate predictions for a sample
        if len(multi_results['successful_drugs']) > 0:
            sample_predictions = multi_predictor.predict_drug_response(
                integrated_features.head(10), drugs=multi_results['successful_drugs']
            )
            
            # Get drug rankings for first sample
            sample_id = sample_predictions.index[0]
            drug_rankings = multi_predictor.get_drug_rankings(sample_id, sample_predictions)
            
            print(f"✓ Drug sensitivity ranking for {sample_id}:")
            for idx, row in drug_rankings.head(5).iterrows():
                print(f"  {row['Sensitivity_Rank']}. {row['Drug']}: {row['Predicted_Response']:.3f}")
        
        # Step 8: Visualization
        print("\n8. Results Visualization")
        print("-" * 40)
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Data overview plot
        omics_data = {
            'Drug Sensitivity': drug_response_matrix,
            'Genomics': genomics_features,
            'Expression': expression_features
        }
        
        plot_data_overview(omics_data, output_path=viz_dir / "data_overview.png")
        print("✓ Generated data overview plot")
        
        # Model performance plot
        model_performances = predictor.get_model_performances()
        plot_model_performance(model_performances, output_path=viz_dir / "model_performance.png")
        print("✓ Generated model performance plot")
        
        # Biomarker analysis plot
        if len(biomarkers) > 0:
            plot_biomarker_analysis(biomarkers, output_path=viz_dir / "biomarker_analysis.png")
            print("✓ Generated biomarker analysis plot")
        
        # Pathway network plot (if pathways found)
        if 'pathway_results' in locals() and len(pathway_results) > 0:
            significant_pathway_ids = pathway_results[pathway_results['p_value'] < 0.05]['pathway_id'].tolist()
            if significant_pathway_ids:
                network_data = pathway_analyzer.get_pathway_network(significant_pathway_ids)
                plot_pathway_network(network_data, output_path=viz_dir / "pathway_network.png")
                print("✓ Generated pathway network plot")
        
        # Step 9: Save Results
        print("\n9. Saving Results")
        print("-" * 40)
        
        # Save biomarkers
        if len(biomarkers) > 0:
            biomarkers.to_csv(output_dir / "discovered_biomarkers.csv", index=False)
            print("✓ Saved biomarkers")
        
        # Save pathway results
        if 'pathway_results' in locals() and len(pathway_results) > 0:
            pathway_results.to_csv(output_dir / "pathway_analysis.csv", index=False)
            print("✓ Saved pathway analysis")
        
        # Save model performance
        performance_df = pd.DataFrame(model_performances).T
        performance_df.to_csv(output_dir / "model_performance.csv")
        print("✓ Saved model performance")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'COSMIC_ID': X_test.index,
            'True_Response': y_test.values,
            'Predicted_Response': predictor.predict(X_test),
            'Drug': target_drug
        })
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        print("✓ Saved predictions")
        
        # Summary report
        summary_report = {
            'target_drug': target_drug,
            'n_samples': len(integrated_features),
            'n_features': integrated_features.shape[1],
            'best_model': predictor.best_model['algorithm'],
            'test_r2': test_metrics['r2_score'],
            'cv_r2_mean': cv_results['mean_metrics']['r2_score'],
            'cv_r2_std': cv_results['std_metrics']['r2_score'],
            'n_biomarkers': len(biomarkers) if len(biomarkers) > 0 else 0,
            'n_significant_pathways': len(significant_pathways) if 'significant_pathways' in locals() else 0
        }
        
        import json
        with open(output_dir / "summary_report.json", 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("✓ Saved summary report")
        
        # Final Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Target Drug: {target_drug}")
        print(f"Best Model: {predictor.best_model['algorithm']}")
        print(f"Test R²: {test_metrics['r2_score']:.4f}")
        print(f"CV R²: {cv_results['mean_metrics']['r2_score']:.4f} ± {cv_results['std_metrics']['r2_score']:.4f}")
        print(f"Biomarkers Discovered: {len(biomarkers) if len(biomarkers) > 0 else 0}")
        print(f"Significant Pathways: {len(significant_pathways) if 'significant_pathways' in locals() else 0}")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
        return summary_report
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return None


if __name__ == "__main__":
    summary = main()
    
    if summary:
        print("\n🎉 Multi-omics drug response prediction analysis completed successfully!")
        print("Check the results directory for detailed outputs and visualizations.")
    else:
        print("\n💥 Analysis failed. Check the logs for details.")
        sys.exit(1)