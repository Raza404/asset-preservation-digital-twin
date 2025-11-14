# backend/scripts/label_flight_log.py

import sys
import os
import pandas as pd
import numpy as np
import argparse

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

# Import all your parsers
from data_ingestion.ardupilot_parser import ArduPilotParser
from data_ingestion.betaflight_parser import BetaflightParser
from data_ingestion.px4_parser import PX4Parser

from ml.feature_engineering import UniversalFeatureExtractor

def get_parser(file_path):
    """Auto-detects the correct parser based on file extension."""
    filename, ext = os.path.splitext(file_path)
    
    if ext.upper() == '.BIN':
        print(f"Detected ArduPilot .bin log: {file_path}")
        return ArduPilotParser(file_path)
    elif ext.upper() == '.CSV':
        print(f"Detected Betaflight .csv log: {file_path}")
        return BetaflightParser(file_path)
    elif ext.upper() == '.ULG':
        print(f"Detected PX4 .ulg log: {file_path}")
        return PX4Parser(file_path)
    elif ext.upper() == '.LOG':
        print(f"Detected .log file, attempting to parse as ArduPilot: {file_path}")
        return ArduPilotParser(file_path)
    else:
        print(f"Error: No parser found for file extension '{ext}'")
        return None

def main(args):
    print("============================================================")
    print("⚙️ FLIGHT LOG LABELING TOOL")
    print("============================================================")

    # 1. Get the correct parser for the file
    parser = get_parser(args.file_path)
    if not parser:
        return

    # 2. Parse the log
    print("\n1️⃣ Parsing flight log...")
    try:
        telemetry_records = parser.parse()
    except Exception as e:
        print(f"--- PARSING FAILED: {e} ---")
        return
        
    if not telemetry_records:
        print("Parsing failed. No records found.")
        return
    print(f"✅ Parsed {len(telemetry_records)} records.")

    # 3. Extract features
    print("\n2️⃣ Extracting universal features...")
    extractor = UniversalFeatureExtractor()
    features_df = extractor.extract_features(telemetry_records)
    print(f"✅ Extracted {features_df.shape[1]} features.")
    
    # 4. Label the data
    print(f"\n3️⃣ Labeling data...")
    
    # Define time column
    time_col = 'time_since_start'
    if time_col not in features_df.columns:
        print(f"Error: '{time_col}' not found in features.")
        return

    # 1. Default all labels to 0 (Normal)
    features_df['failure_type'] = 0
    
    # 2. Find the start of the "imminent failure" warning window
    warning_start_time = args.failure_time - args.window_size
    
    # 3. Mark all data points inside this window with the specified label
    imminent_indices = features_df[
        (features_df[time_col] >= warning_start_time) & 
        (features_df[time_col] < args.failure_time)
    ].index
    
    if len(imminent_indices) == 0:
        print(f"Warning: No data points found in the warning window ({warning_start_time}s to {args.failure_time}s).")
    else:
        features_df.loc[imminent_indices, 'failure_type'] = args.failure_type
        print(f"   Labeled {len(imminent_indices)} records as 'Type {args.failure_type}'")

    # 4. Trim all data at and after the failure event
    original_count = len(features_df)
    features_df = features_df[features_df[time_col] < args.failure_time].copy()
    print(f"   Trimmed {original_count - len(features_df)} post-failure records.")

    # 5. Save the labeled data
    # Create the processed directory if it doesn't exist
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create a unique name for the output file
    base_name = os.path.basename(args.file_path)
    output_filename = f"labeled_{base_name}.csv"
    output_path = os.path.join(processed_dir, output_filename)
    
    # Clean up NaNs before saving
    features_df = features_df.bfill().ffill().fillna(0)
    
    features_df.to_csv(output_path, index=False)
    
    print("\n--- ✅ RESULTS ---")
    print(f"Successfully generated labeled dataset!")
    print(f"Saved to: {output_path}")
    print("\nData balance in this file:")
    print(features_df['failure_type'].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeling tool for flight log failure analysis.")
    parser.add_argument("file_path", type=str, help="Path to the flight log file (e.g., .bin, .csv)")
    parser.add_argument("--failure_type", type=int, default=1, help="Integer label for this failure (e.g., 1 for MotorFail, 2 for GPSFail)")
    parser.add_argument("--failure_time", type=float, required=True, help="The exact timestamp (in seconds) when the failure event begins.")
    parser.add_argument("--window_size", type=float, default=2.0, help="The duration (in seconds) *before* the failure_time to label as 'imminent'.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        sys.exit(1)
        
    main(args)