#!/usr/bin/env python3
"""
Simple test to verify the multi-omics package works correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic imports...")
    
    try:
        # Test core scientific packages
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        print("✓ Core scientific packages imported successfully")
        
        # Test machine learning packages
        import xgboost as xgb
        import lightgbm as lgb
        print("✓ Machine learning packages imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_synthetic_data_processing():
    """Test basic data processing with synthetic data."""
    print("\nTesting synthetic data processing...")
    
    try:
        # Create synthetic multi-omics data
        np.random.seed(42)
        n_samples, n_genes = 100, 1000
        
        # Synthetic gene expression data
        expression_data = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            columns=[f"gene_{i}" for i in range(n_genes)],
            index=[f"sample_{i}" for i in range(n_samples)]
        )
        
        # Synthetic drug response data
        drug_response = pd.Series(
            np.random.randn(n_samples),
            index=[f"sample_{i}" for i in range(n_samples)],
            name="drug_response"
        )
        
        print(f"✓ Created synthetic expression data: {expression_data.shape}")
        print(f"✓ Created synthetic drug response data: {drug_response.shape}")
        
        # Basic preprocessing
        expression_normalized = (expression_data - expression_data.mean()) / expression_data.std()
        print("✓ Normalized expression data")
        
        # Simple correlation analysis
        correlations = expression_normalized.corrwith(drug_response)
        top_genes = correlations.abs().nlargest(10)
        print(f"✓ Found top 10 correlated genes: {len(top_genes)}")
        
        return True
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def test_machine_learning():
    """Test basic machine learning functionality."""
    print("\nTesting machine learning...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 200, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ Model trained successfully")
        print(f"✓ MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Machine learning error: {e}")
        return False

def test_xgboost():
    """Test XGBoost functionality."""
    print("\nTesting XGBoost...")
    
    try:
        import xgboost as xgb
        
        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 100, 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Create XGBoost model
        model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        print(f"✓ XGBoost model trained and predictions made: {len(predictions)}")
        
        return True
    except Exception as e:
        print(f"❌ XGBoost error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Multi-omics Biomarker Discovery - Simple Test")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Synthetic Data Processing", test_synthetic_data_processing),
        ("Machine Learning", test_machine_learning),
        ("XGBoost", test_xgboost)
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
        print("🎉 All tests passed! The environment is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)