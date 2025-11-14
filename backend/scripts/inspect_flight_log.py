# backend/scripts/inspect_flight_log.py

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

# Import all your parsers
from data_ingestion.ardupilot_parser import ArduPilotParser
from data_ingestion.betaflight_parser import BetaflightParser
from data_ingestion.px4_parser import PX4Parser

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

def main(file_path):
    print("============================================================")
    print("✈️ FLIGHT LOG FORENSICS INSPECTOR (RAW DATA)")
    print("============================================================")

    parser = get_parser(file_path)
    if not parser:
        return

    print("\n1️⃣ Parsing flight log...")
    try:
        telemetry_records = parser.parse()
    except Exception as e:
        print(f"--- PARSING FAILED ---")
        print(f"Error: {e}")
        return
        
    if not telemetry_records:
        print("Parsing failed. No records found.")
        return
    print(f"✅ Parsed {len(telemetry_records)} records.")

    print("\n2️⃣ Converting raw records to DataFrame...")
    raw_df = pd.DataFrame([r.__dict__ for r in telemetry_records])
    
    # --- NEW: FULL DIAGNOSTIC PRINT ---
    # This will show us every column name AND a statistical summary
    print(f"✅ DataFrame created with {len(raw_df.columns)} raw columns. Available columns are:")
    print(list(raw_df.columns))
    print("\n--- DATA SUMMARY (STATISTICS) ---")
    print(raw_df.describe())
    # ==================================
    
    time_col = 'time_since_start'
    if 'timestamp' in raw_df.columns:
        try:
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], unit='s', errors='coerce')
            raw_df[time_col] = (raw_df['timestamp'] - raw_df['timestamp'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Could not convert timestamp: {e}")
            time_col = 'timestamp' 
    else:
        print("Error: Could not find a time column.")
        return

    g_force_col = 'g_force'
    accel_cols_found = True
    for col in ['accel_x', 'accel_y', 'accel_z']:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        else:
            accel_cols_found = False

    if accel_cols_found:
         raw_df[g_force_col] = np.sqrt(raw_df['accel_x']**2 + raw_df['accel_y']**2 + raw_df['accel_z']**2) / 9.81
    else:
        g_force_col = None 
        print("Warning: Standard 'accel_x/y/z' columns not found. Cannot plot G-Force.")

    print("\n3️⃣ Generating diagnostic plots...")
    
    def find_col(df, options):
        """Robustly find a column by checking a list of possible names."""
        for opt in options:
            if opt in df.columns:
                print(f"   Found column: '{opt}'")
                return opt
        return None

    roll_col = find_col(raw_df, ['roll', 'Roll', 'ATT.Roll', 'ATT_Roll'])
    pitch_col = find_col(raw_df, ['pitch', 'Pitch', 'ATT.Pitch', 'ATT_Pitch'])
    
    gyro_x_col = find_col(raw_df, ['gyro_x', 'GyrX', 'IMU.GyrX', 'IMU_GyrX'])
    gyro_y_col = find_col(raw_df, ['gyro_y', 'GyrY', 'IMU.GyrY', 'IMU_GyrY'])
    gyro_z_col = find_col(raw_df, ['gyro_z', 'GyrZ', 'IMU.GyrZ', 'IMU_GyrZ'])

    motor_cols = [col for col in raw_df.columns if 'motor_' in col.lower() or col.startswith('RCOU.C') or col.startswith('RCOU_C')]
    
    vib_cols = [col for col in raw_df.columns if 'vibration' in col.lower() or col.startswith('VIBE_') or col.startswith('VIBE.')]

    fig, axes = plt.subplots(5, 1, figsize=(20, 25), sharex=True)
    fig.suptitle(f'Forensics Report: {os.path.basename(file_path)}', fontsize=20)

    # Plot 1: G-Force
    if g_force_col and raw_df[g_force_col].count() > 0:
        axes[0].plot(raw_df[time_col], raw_df[g_force_col], label='G-Force (Calculated)', color='r')
        axes[0].set_title('Total G-Force', fontsize=16)
        axes[0].set_ylabel('G')
        axes[0].grid(True)
    else:
        axes[0].set_title('G-Force (Not Found or Empty)', fontsize=16)

    # Plot 2: Attitude (Roll/Pitch)
    if roll_col and pitch_col and raw_df[roll_col].count() > 0:
        axes[1].plot(raw_df[time_col], raw_df[roll_col], label=roll_col)
        axes[1].plot(raw_df[time_col], raw_df[pitch_col], label=pitch_col)
        axes[1].set_title('Attitude (Roll & Pitch)', fontsize=16)
        axes[1].set_ylabel('Degrees')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_title('Attitude (Not Found or Empty)', fontsize=16)
        
    # Plot 3: Gyro Rates
    if gyro_x_col and gyro_y_col and gyro_z_col and raw_df[gyro_x_col].count() > 0:
        axes[2].plot(raw_df[time_col], raw_df[gyro_x_col], label=gyro_x_col)
        axes[2].plot(raw_df[time_col], raw_df[gyro_y_col], label=gyro_y_col)
        axes[2].plot(raw_df[time_col], raw_df[gyro_z_col], label=gyro_z_col)
        axes[2].set_title('Gyroscope Rates (Rotation)', fontsize=16)
        axes[2].set_ylabel('deg/s')
        axes[2].legend()
        axes[2].grid(True)
    else:
         axes[2].set_title('Gyroscope (Not Found or Empty)', fontsize=16)

    # Plot 4: Motor Outputs
    if motor_cols and raw_df[motor_cols[0]].count() > 0:
        print(f"   Found motor columns: {motor_cols}")
        for col in motor_cols:
            axes[3].plot(raw_df[time_col], raw_df[col], label=col)
        axes[3].set_title('Motor Outputs', fontsize=16)
        axes[3].set_ylabel('PWM / Throttle')
        axes[3].legend()
        axes[3].grid(True)
    else:
         axes[3].set_title('Motor Outputs (Not Found or Empty)', fontsize=16)

    # Plot 5: Vibration
    if vib_cols and raw_df[vib_cols[0]].count() > 0:
         print(f"   Found vibration columns: {vib_cols}")
         for col in vib_cols:
            axes[4].plot(raw_df[time_col], raw_df[col], label=col)
         axes[4].set_title('Vibration Levels', fontsize=16)
         axes[4].set_ylabel('m/s^2')
         axes[4].legend()
         axes[4].grid(True)
    else:
        axes[4].set_title('Vibration (Not Found or Empty)', fontsize=16)

    axes[-1].set_xlabel('Time (seconds)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f"inspection_report_{os.path.basename(file_path)}.png"
    output_path = os.path.join(project_root, 'data', 'validation', output_filename)
    plt.savefig(output_path)
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_flight_log.py <path_to_log_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    main(file_path)
