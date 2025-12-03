#!/usr/bin/env python3
"""
Fix model pickle for Kaggle compatibility by removing NumPy random state.
This resolves the PCG64 BitGenerator compatibility issue.
"""

import joblib
import sys
from pathlib import Path


def fix_model_random_state(input_path, output_path=None):
    """
    Load a model and remove any random_state references that cause
    NumPy version compatibility issues.
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading model from: {input_path}")
    meta = joblib.load(input_path)
    
    print(f"Model contains keys: {list(meta.keys())}")
    
    # Fix random state in the model objects
    for model_name, model in meta.get('models', {}).items():
        if hasattr(model, 'random_state'):
            # Replace with a simple integer seed instead of RandomState object
            print(f"Fixing random_state in {model_name} model")
            model.random_state = 42
    
    # Remove any other random state references in best_params
    if 'best_params' in meta:
        for params_key, params in meta['best_params'].items():
            if 'random_state' in params:
                print(f"Fixing random_state in {params_key} params")
                params['random_state'] = 42
    
    print(f"Saving fixed model to: {output_path}")
    joblib.dump(meta, output_path)
    print("✓ Model fixed successfully")
    
    # Verify it loads
    print("Verifying fixed model loads...")
    test_meta = joblib.load(output_path)
    print(f"✓ Verification successful - loaded {len(test_meta.get('feature_columns', []))} features")


if __name__ == '__main__':
    input_model = sys.argv[1] if len(sys.argv) > 1 else 'models/best_model.pkl'
    output_model = sys.argv[2] if len(sys.argv) > 2 else input_model
    
    fix_model_random_state(input_model, output_model)
