# backend/scripts/detect_anomalies.py

import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

from data_ingestion.betaflight_parser import BetaflightParser
from ml.feature_engineering import UniversalFeatureExtractor 

def main(file_path):
    print("============================================================")
    print("🎯 ANOMALY DETECTION (PRO) - Now with Universal Features")
    print("============================================================")

    # 1. Parse the flight data
    print(f"\n1️⃣ Parsing flight log: {os.path.basename(file_path)}...")
    parser = BetaflightParser(file_path)
    telemetry_records = parser.parse() # This is the list of objects
    if not telemetry_records:
        print("Parsing failed. Exiting.")
        return
    
    # We create this DataFrame for later, but don't pass it to the extractor
    telemetry_df = pd.DataFrame([record.__dict__ for record in telemetry_records])
    print(f"✅ Parsed {len(telemetry_df)} records.")

    # 2. Extract features using the REAL extractor
    print("\n2️⃣ Extracting features with UniversalFeatureExtractor...")
    
    extractor = UniversalFeatureExtractor()
    
    # === FIX IS HERE ===
    # Pass the original 'telemetry_records' list, not the 'telemetry_df'
    features_df = extractor.extract_features(telemetry_records)
    # ===================

    print(f"✅ Extracted {features_df.shape[1]} total features.")
    
    # 3. Select a rich set of model features
    model_features = [
        'g_force',
        'jerk_magnitude',
        'vibration_x_rolling_mean',
        'vibration_y_rolling_mean',
        'vibration_z_rolling_mean',
        'jerk_magnitude_rolling_std',
        'g_force_rolling_max',
        'rotation_rate_magnitude' 
    ]
    
    available_features = [f for f in model_features if f in features_df.columns]
    
    if 'time_since_start' not in available_features:
         if 'time_since_start' in features_df.columns:
              available_features.append('time_since_start')
    
    print(f"   Using {len(available_features)} features for detection: {available_features}")

    model_df = features_df[available_features].copy()
    model_df = model_df.bfill().ffill() # Fill NaNs from rolling windows
    model_df = model_df.fillna(0) # Fill any remaining NaNs

    X = model_df.drop(columns=['time_since_start'], errors='ignore')

    # 4. Scale the features
    print("\n3️⃣ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train the Isolation Forest model
    print("\n4️⃣ Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=0.0005, random_state=42)
    model.fit(X_scaled)

    # 6. Predict anomalies and display results
    print("\n5️⃣ Predicting anomalies and analyzing results...")
    # Make sure X_scaled has the same index as features_df
    X_scaled_df = pd.DataFrame(X_scaled, index=features_df.index, columns=X.columns)
    
    features_df['anomaly_score'] = model.decision_function(X_scaled_df)
    features_df['is_anomaly'] = model.predict(X_scaled_df)

    anomalies = features_df[features_df['is_anomaly'] == -1].sort_values(by='g_force', ascending=False)

    print("\n--- 📈 RESULTS ---")
    print(f"Total anomalies detected: {len(anomalies)}")

    if not anomalies.empty:
        print("Top 5 most significant anomalous events (sorted by G-force):")
        display_cols = ['time_since_start', 'g_force'] + [f for f in available_features if f not in ['time_since_start', 'g_force']]
        print(anomalies[display_cols].head(5).to_string())
    else:
        print("No significant anomalies were detected with the current settings.")
    
    print("\n✅ Anomaly detection complete!")


if __name__ == "__main__":
    default_path = os.path.join(project_root, 'data', 'public', 'betaflight_logs', 'my_flight_001.csv')
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print(f"No file path provided. Running with default:\n{default_path}\n")
        main(default_path)