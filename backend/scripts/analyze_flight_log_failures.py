"""
Real Flight Log Failure Detection
Analyzes actual flight data to detect potential failures using rule-based system.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.rule_based_predictor import RuleBasedPredictor


def load_flight_log(csv_path: str) -> pd.DataFrame:
    """Load flight log CSV."""
    print(f"\n📂 Loading flight log: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} telemetry records")
    print(f"✓ Duration: {(len(df) * 0.1):.1f} seconds")
    print(f"✓ Columns: {', '.join(df.columns[:10])}...")
    return df


def prepare_telemetry_buffer(df: pd.DataFrame, start_idx: int, window_size: int = 30) -> list:
    """Extract telemetry window from dataframe."""
    end_idx = min(start_idx + window_size, len(df))
    window_df = df.iloc[start_idx:end_idx]
    
    telemetry_buffer = []
    for _, row in window_df.iterrows():
        # Map CSV columns to expected telemetry format
        telemetry = {
            'battery_voltage': row.get('battery_voltage', 12.0),
            'battery_remaining': row.get('battery_remaining', 100),
            'throttle': calculate_throttle_from_motors(row),
            'ground_speed': row.get('ground_speed', 0.0),
            'altitude': row.get('altitude', 0.0),
            'motor_temp': calculate_avg_motor_temp(row),
            'esc_temp': 50.0,  # Not in this dataset, use default
            'vibration_magnitude': calculate_vibration_magnitude(row),
            'gps_satellites': 12,  # Not in this dataset, assume good
            'gyro_x': row.get('gyro_x', 0.0),
            'gyro_y': row.get('gyro_y', 0.0),
            'gyro_z': row.get('gyro_z', 0.0),
        }
        telemetry_buffer.append(telemetry)
    
    return telemetry_buffer


def calculate_throttle_from_motors(row) -> float:
    """Calculate average throttle from motor RPMs."""
    rpms = []
    for i in range(1, 5):
        rpm = row.get(f'motor_{i}_rpm', 0)
        if rpm > 0:
            rpms.append(rpm)
    
    if rpms:
        avg_rpm = np.mean(rpms)
        # Normalize RPM to throttle % (assuming 6000 RPM = 100%)
        throttle = (avg_rpm / 6000.0) * 100.0
        return min(100.0, max(0.0, throttle))
    return 50.0


def calculate_avg_motor_temp(row) -> float:
    """Calculate average motor temperature."""
    temps = []
    for i in range(1, 5):
        temp = row.get(f'motor_{i}_temp', None)
        if temp is not None:
            temps.append(temp)
    
    return np.mean(temps) if temps else 50.0


def calculate_vibration_magnitude(row) -> float:
    """Calculate vibration magnitude from x, y, z components."""
    vib_x = row.get('vibration_x', 0.0)
    vib_y = row.get('vibration_y', 0.0)
    vib_z = row.get('vibration_z', 0.0)
    
    return np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)


def print_failure_alert(timestamp, component, health, sample_idx, total_samples):
    """Print formatted failure alert."""
    status_emoji = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'CRITICAL': '🚨'
    }
    
    emoji = status_emoji.get(health.status, '❓')
    progress = (sample_idx / total_samples) * 100
    
    print(f"\n{emoji} [{progress:5.1f}%] Time: {timestamp}")
    print(f"    Component: {component.upper()}")
    print(f"    Status: {health.status} | Risk: {health.risk_level:.1%} | RUL: {health.predicted_rul}s")
    
    if health.status != 'HEALTHY':
        print(f"    Triggers:")
        for trigger in health.triggers[:3]:  # Show top 3
            print(f"      • {trigger.replace('_', ' ').title()}")
        
        print(f"    Recommendations:")
        for rec in health.recommendations[:2]:  # Show top 2
            print(f"      → {rec}")


def analyze_flight_log(csv_path: str, analysis_interval: int = 100):
    """
    Analyze flight log for potential failures.
    
    Args:
        csv_path: Path to flight log CSV
        analysis_interval: Analyze every N samples (e.g., 100 = every 10 seconds)
    """
    print("="*70)
    print("REAL FLIGHT LOG FAILURE DETECTION")
    print("="*70)
    
    # Load data
    df = load_flight_log(csv_path)
    
    # Initialize predictor
    predictor = RuleBasedPredictor()
    print("\n🔍 Initializing rule-based failure predictor...")
    
    # Track failures over time
    failure_timeline = []
    warning_count = 0
    critical_count = 0
    
    # Analyze at intervals
    print(f"\n⏳ Analyzing flight (every {analysis_interval} samples = {analysis_interval * 0.1:.1f}s)...\n")
    
    for idx in range(0, len(df), analysis_interval):
        # Get 30-second window
        telemetry_buffer = prepare_telemetry_buffer(df, idx, window_size=30)
        
        if len(telemetry_buffer) < 10:
            continue
        
        # Analyze
        results = predictor.analyze_telemetry(telemetry_buffer)
        
        # Get timestamp
        timestamp = df.iloc[idx]['timestamp'] if 'timestamp' in df.columns else f"Sample {idx}"
        
        # Check for issues
        has_warning = False
        has_critical = False
        
        for comp_name, health in results.items():
            if health.status == 'CRITICAL':
                critical_count += 1
                has_critical = True
                print_failure_alert(timestamp, comp_name, health, idx, len(df))
            elif health.status == 'WARNING':
                warning_count += 1
                has_warning = True
                # Only print warnings if no critical
                if not has_critical:
                    print_failure_alert(timestamp, comp_name, health, idx, len(df))
        
        # Record timeline
        max_risk = max([h.risk_level for h in results.values()])
        failure_timeline.append({
            'timestamp': timestamp,
            'sample_idx': idx,
            'max_risk': max_risk,
            'has_warning': has_warning,
            'has_critical': has_critical
        })
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📊 Flight Summary:")
    print(f"  • Total samples analyzed: {len(failure_timeline)}")
    print(f"  • Warnings detected: {warning_count}")
    print(f"  • Critical alerts: {critical_count}")
    
    # Risk timeline
    print(f"\n📈 Risk Timeline:")
    for entry in failure_timeline:
        status_str = "🚨 CRITICAL" if entry['has_critical'] else "⚠️  WARNING" if entry['has_warning'] else "✅ HEALTHY"
        print(f"  {entry['timestamp'][:19]:19s} | Risk: {entry['max_risk']:5.1%} | {status_str}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if critical_count > 0:
        print(f"  🚨 CRITICAL: Flight had {critical_count} critical failure alerts!")
        print(f"     → Immediate landing would have been recommended")
    elif warning_count > 0:
        print(f"  ⚠️  WARNING: Flight had {warning_count} warning alerts")
        print(f"     → Reduced operations and early landing recommended")
    else:
        print(f"  ✅ HEALTHY: No significant failures detected")
        print(f"     → Flight parameters remained within safe limits")
    
    return failure_timeline


def main():
    # Analyze the sample flight log
    csv_path = "data/raw/sample_flight_log.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ Flight log not found: {csv_path}")
        return
    
    analyze_flight_log(csv_path, analysis_interval=200)  # Analyze every 20 seconds
    
    print("\n" + "="*70)
    print("✓ Failure detection complete!")
    print("="*70)


if __name__ == "__main__":
    main()
