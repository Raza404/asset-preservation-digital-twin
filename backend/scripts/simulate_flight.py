# backend/scripts/simulate_flight.py

import sys
import os
import pandas as pd
import numpy as np
import joblib
import time
import warnings # <-- This is the only import we need

# --- FIX #1: Disable the UserWarning ---
# We just call the built-in UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
# --------------------------------------

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

from data_ingestion.ardupilot_parser import ArduPilotParser
from ml.feature_engineering import UniversalFeatureExtractor

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(project_root, 'backend', 'src', 'ml', 'failure_model.joblib')
EXTRACTOR = UniversalFeatureExtractor()

# Define our "stress" thresholds
VIBRATION_WARN_THRESHOLD = 20.0
JERK_WARN_THRESHOLD = 50.0

def load_model(path):
    """Loads the pre-trained failure model."""
    print(f"Loading model from {path}...")
    if not os.path.exists(path):
        print(f"FATAL: Model file not found at {path}")
        return None
    model = joblib.load(path)
    print("✅ Model loaded.")
    return model

def get_guidance(model, feature_df_row):
    """
    The "brain" of the guidance system.
    Takes a single-row DataFrame and returns a plain-English suggestion.
    """
    prediction = model.predict(feature_df_row)[0]
    
    # 2. Check for CRITICAL failures
    if prediction == 1:
        return "🔴 CRITICAL! Motor Failure signature detected. LAND IMMEDIATELY."
    if prediction == 2:
        return "🔴 CRITICAL! High-Vibration Failure signature detected. LAND IMMEDIATELY."
        
    # 3. If no critical failure, check for preventative suggestions
    vib_mag = feature_df_row['vibration_x_rolling_mean'].iloc[0]
    jerk_mag = feature_df_row['jerk_magnitude_rolling_mean'].iloc[0]

    if vib_mag > VIBRATION_WARN_THRESHOLD:
        return f"🟡 WARNING: High vibration ({vib_mag:.1f}). Reduce throttle to decrease component stress."
    
    if jerk_mag > JERK_WARN_THRESHOLD:
        return f"🟡 WARNING: High G-force maneuver ({jerk_mag:.1f}). Fly smoother to reduce fatigue."

    # 4. If all checks pass
    return "✅ NORMAL. All systems nominal."


def main(log_file_path):
    print("============================================================")
    print("🛰️  ASSET PRESERVATION - FLIGHT SIMULATOR")
    print("============================================================")

    # 1. Load the trained AI model
    model = load_model(MODEL_PATH)
    if model is None:
        return
        
    model_features = model.feature_names_in_
    print(f"Model requires {len(model_features)} features.")

    # 2. Load and PRE-PROCESS the *entire* log file
    print(f"\nLoading and pre-processing log: {os.path.basename(log_file_path)}...")
    parser = ArduPilotParser(log_file_path)
    telemetry_records = parser.parse()
    if not telemetry_records:
        print("Failed to parse log file.")
        return
        
    print("Extracting features from entire log (this matches training)...")
    features_df = EXTRACTOR.extract_features(telemetry_records)
    features_df = features_df.bfill().ffill().fillna(0)
    
    # Ensure the DataFrame has the exact columns the model expects
    features_for_model = features_df.reindex(columns=model_features, fill_value=0)
    
    # Get the time column for our dashboard
    time_col = features_df['time_since_start']
    
    print(f"✅ Log pre-processed. Simulating {len(features_for_model)} records...")
    print("--- Press [Ctrl+C] to stop ---")
    time.sleep(2)

    # 3. Simulate the "live" flight by looping through the pre-processed features
    try:
        for i in range(len(features_for_model)):
            current_features_df = features_for_model.iloc[[i]] 
            current_time = time_col.iloc[i]
            
            # Get the guidance suggestion
            suggestion = get_guidance(model, current_features_df)
            
            # Print the "live" dashboard
            print(f"[T={current_time:.2f}s] {suggestion}          ", end='\r')

    except KeyboardInterrupt:
        print("\n--- Simulation Stopped by User ---")
        return

    print("\n--- Simulation Complete ---")


if __name__ == "__main__":
    # Test this on one of our known crash logs
    test_log = os.path.join(project_root, 'data', 'failures', 'case_001_ardupilot_crash', '2022-07-11 15-22-01.bin')
    
    if not os.path.exists(test_log):
        print(f"Error: Test log not found at {test_log}")
    else:
        main(test_log)