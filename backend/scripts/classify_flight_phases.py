# backend/scripts/classify_flight_phases.py

import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

from data_ingestion.betaflight_parser import BetaflightParser
# --- 1. Import your REAL feature extractor ---
from ml.feature_engineering import UniversalFeatureExtractor 

# --- 2. The old placeholder function is GONE ---


def main(file_path):
    print("============================================================")
    print("✈️ FLIGHT PHASE CLASSIFICATION (PRO) - Now with Universal Features")
    print("============================================================")

    # 1. Load data
    print(f"\n1️⃣ Parsing flight log: {os.path.basename(file_path)}...")
    parser = BetaflightParser(file_path)
    telemetry_records = parser.parse() # Pass the list of objects
    if not telemetry_records:
        print("Parsing failed. Exiting.")
        return
    print(f"✅ Parsed {len(telemetry_records)} records.")

    # 2. Extract features
    print("\n2️⃣ Extracting features with UniversalFeatureExtractor...")
    extractor = UniversalFeatureExtractor()
    features_df = extractor.extract_features(telemetry_records)
    print(f"✅ Extracted {features_df.shape[1]} total features.")
    
    # 3. Select features for clustering
    # We can now use a much richer set of features
    cluster_features = [
        'g_force_rolling_mean',
        'ground_speed',
        'vertical_speed',
        'jerk_magnitude_rolling_mean',
        'vibration_magnitude_rolling_mean', # Assuming you have a vibration magnitude
        'rotation_rate_magnitude'
    ]
    
    # Let's dynamically find available features
    available_features = [f for f in cluster_features if f in features_df.columns]
    
    # Fallback if specific rolling means aren't present
    if not available_features:
        available_features = ['g_force', 'ground_speed', 'vertical_speed', 'roll', 'pitch']
        available_features = [f for f in available_features if f in features_df.columns]

    print(f"   Using {len(available_features)} features for clustering: {available_features}")

    # Prepare data for clustering
    X = features_df[available_features].copy()
    X = X.bfill().ffill() # Handle NaNs from rolling windows
    X = X.fillna(0) # Handle any remaining NaNs

    # 4. Scale features and apply K-Means
    print("\n3️⃣ Scaling features and clustering with K-Means...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Let's try 5 clusters now that we have richer data
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    features_df['flight_phase'] = kmeans.fit_predict(X_scaled)
    print("✅ Clustering complete.")

    # 5. Analyze and visualize the results
    print("\n4️⃣ Analyzing and visualizing clusters...")
    phase_summary = features_df.groupby('flight_phase')[available_features].mean().sort_values(by='g_force' if 'g_force' in available_features else available_features[0])
    print("\n--- Cluster Characteristics (Mean Values) ---")
    print(phase_summary.to_string())

    # 6. Generate plot
    print("\n5️⃣ Generating plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    
    scatter = ax.scatter(features_df['time_since_start'], features_df['g_force'], c=features_df['flight_phase'], cmap='viridis', s=5, alpha=0.7)
    
    legend_handles, _ = scatter.legend_elements()
    phase_labels = [f"Phase {i}" for i in range(kmeans.n_clusters)]
    ax.legend(legend_handles, phase_labels, title="Flight Phases", bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.set_title(f'Flight Phases Identified by Universal Features', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('G-Force', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    
    output_path = os.path.join(project_root, 'data', 'validation', 'flight_phase_classification_pro.png')
    plt.savefig(output_path)
    print(f"\n✅ Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    default_path = os.path.join(project_root, 'data', 'public', 'betaflight_logs', 'my_flight_001.csv')
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(default_path)