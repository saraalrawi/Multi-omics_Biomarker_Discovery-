#!/usr/bin/env python3
"""
Simple test script to verify the Multi-omics Biomarker Discovery project works.
Run this to test the basic functionality.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data_acquisition.gdsc import GDSCDataAcquisition
        print("✓ Data acquisition module imported")
        
        from preprocessing.multiomics import MultiOmicsPreprocessor
        print("✓ Preprocessing module imported")
        
        from modeling import DrugResponsePredictor, DrugResponseEvaluator
        print("✓ Modeling modules imported")
        
        from biomarker_discovery import BiomarkerDiscovery
        print("✓ Biomarker discovery module imported")
        
        from pathway_analysis import PathwayAnalyzer
        print("✓ Pathway analysis module imported")
        
        from visualization import plot_data_overview
        print("✓ Visualization module imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_acquisition():
    """Test GDSC data acquisition."""
    print("\nTesting GDSC data acquisition...")
    
    try:
        from data_acquisition.gdsc import GDSCDataAcquisition
        
        # Create data directory
        data_dir = project_root / "data" / "gdsc_cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GDSC data acquisition
        gdsc = GDSCDataAcquisition()
        
        # Download sample data
        success = gdsc.download(data_types=["drug_sensitivity", "genomics", "transcriptomics", "cell_lines"])
        
        if success:
            print("✓ GDSC data generated successfully")
            
            # Validate data
            validation = gdsc.validate()
            if validation:
                print("✓ Data validation passed")
                return True
            else:
                print("❌ Data validation failed")
                return False
        else:
            print("❌ Data generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Data acquisition error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality."""
    print("\nTesting preprocessing...")
    
    try:
        from preprocessing.multiomics import MultiOmicsPreprocessor
        
        # Load generated data
        data_dir = project_root / "data" / "gdsc_cache"
        
        drug_sensitivity = pd.read_csv(data_dir / "drug_sensitivity_processed.csv")
        mutations = pd.read_csv(data_dir / "mutations_processed.csv")
        cnv = pd.read_csv(data_dir / "cnv_processed.csv")
        expression = pd.read_csv(data_dir / "expression_processed.csv")
        
        print(f"✓ Loaded data: Drug sensitivity {drug_sensitivity.shape}, Mutations {mutations.shape}")
        
        # Initialize preprocessor
        preprocessor = MultiOmicsPreprocessor()
        
        # Test drug response preprocessing
        drug_matrix, filtered_data = preprocessor.preprocess_drug_response_data(drug_sensitivity)
        print(f"✓ Drug response matrix: {drug_matrix.shape}")
        
        # Test genomics preprocessing
        genomics_features = preprocessor.preprocess_gdsc_genomics(mutations, cnv)
        print(f"✓ Genomics features: {genomics_features.shape}")
        
        # Test expression preprocessing
        expression_features = preprocessor.preprocess_gdsc_expression(expression)
        print(f"✓ Expression features: {expression_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def test_modeling():
    """Test basic modeling functionality."""
    print("\nTesting modeling...")
    
    try:
        from modeling import DrugResponsePredictor
        from sklearn.model_selection import train_test_split
        
        # Create simple test data
        np.random.seed(42)
        n_samples, n_features = 100, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train predictor
        predictor = DrugResponsePredictor(algorithms=["ridge", "random_forest"])
        predictor.fit(X_train, y_train, optimize_hyperparameters=False)  # Skip optimization for speed
        
        # Make predictions
        predictions = predictor.predict(X_test)
        
        print(f"✓ Model trained and predictions made: {len(predictions)} predictions")
        print(f"✓ Best model: {predictor.best_model['algorithm']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Modeling error: {e}")
        return False

def test_biomarker_discovery():
    """Test biomarker discovery."""
    print("\nTesting biomarker discovery...")
    
    try:
        from biomarker_discovery import BiomarkerDiscovery
        
        # Create test data
        np.random.seed(42)
        n_samples, n_features = 100, 200
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        X.columns = [f"feature_{i}" for i in range(n_features)]
        y = pd.Series(np.random.randn(n_samples))
        
        # Initialize biomarker discovery
        biomarker_discovery = BiomarkerDiscovery(feature_selection_method="univariate")
        
        # Discover biomarkers
        biomarkers = biomarker_discovery.discover_biomarkers(X, y)
        
        print(f"✓ Biomarkers discovered: {len(biomarkers)}")
        
        if len(biomarkers) > 0:
            print(f"✓ Top biomarker: {biomarkers.iloc[0]['feature']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Biomarker discovery error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Multi-omics Biomarker Discovery - Project Test")
    print("="*60)
    
    # Create results directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Acquisition Test", test_data_acquisition),
        ("Preprocessing Test", test_preprocessing),
        ("Modeling Test", test_modeling),
        ("Biomarker Discovery Test", test_biomarker_discovery)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! The project is working correctly.")
        print("\nNext steps:")
        print("1. Run the full example: python examples/gdsc_drug_response_prediction.py")
        print("2. Check results in the 'results/' directory")
        return True
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)