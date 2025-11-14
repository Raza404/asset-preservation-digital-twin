import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_flight_data(duration_minutes=10, frequency_hz=10):
    """Generate realistic sample flight data"""
    
    print("="*60)
    print("🔧 GENERATING SAMPLE FLIGHT DATA")
    print("="*60)
    
    # Calculate number of samples
    total_seconds = duration_minutes * 60
    num_samples = total_seconds * frequency_hz
    
    print(f"\nDuration: {duration_minutes} minutes")
    print(f"Frequency: {frequency_hz} Hz")
    print(f"Total samples: {num_samples}")
    
    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i/frequency_hz) for i in range(num_samples)]
    
    # Generate flight profile (takeoff, cruise, landing)
    t = np.linspace(0, total_seconds, num_samples)
    
    # Altitude profile
    altitude = np.zeros(num_samples)
    cruise_alt = 50  # meters
    
    # Takeoff phase (0-60s)
    takeoff_mask = t < 60
    altitude[takeoff_mask] = (t[takeoff_mask] / 60) * cruise_alt
    
    # Cruise phase
    cruise_mask = (t >= 60) & (t < total_seconds - 60)
    altitude[cruise_mask] = cruise_alt + np.random.normal(0, 2, np.sum(cruise_mask))
    
    # Landing phase
    landing_mask = t >= total_seconds - 60
    time_from_landing_start = t[landing_mask] - (total_seconds - 60)
    altitude[landing_mask] = cruise_alt * (1 - time_from_landing_start / 60)
    
    # GPS coordinates (circular pattern)
    center_lat = 37.7749
    center_lon = -122.4194
    radius = 0.001  # degrees
    
    angle = 2 * np.pi * t / total_seconds
    latitude = center_lat + radius * np.cos(angle)
    longitude = center_lon + radius * np.sin(angle)
    
    # IMU data (accelerometer - m/s²)
    accel_x = np.random.normal(0, 0.5, num_samples)
    accel_y = np.random.normal(0, 0.5, num_samples)
    accel_z = np.random.normal(9.81, 1.0, num_samples)  # Gravity + vibration
    
    # Gyroscope (deg/s)
    gyro_x = np.random.normal(0, 5, num_samples)
    gyro_y = np.random.normal(0, 5, num_samples)
    gyro_z = np.random.normal(0, 10, num_samples)
    
    # Battery (draining from 100% to 70%)
    battery_pct = 100 - (30 * t / total_seconds) + np.random.normal(0, 0.5, num_samples)
    battery_voltage = 12.8 + (battery_pct / 100) * 4.0  # 4S LiPo
    
    # Speed
    speed = np.random.normal(5, 1, num_samples)  # m/s
    
    # Create DataFrame
    data = {
        'timestamp': timestamps,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'ground_speed': speed,
        'battery_voltage': battery_voltage,
        'battery_remaining': battery_pct,
        'wind_speed': np.random.normal(2, 0.5, num_samples),
        'air_temperature': np.random.normal(25, 2, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to file - FIXED PATH
    output_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sample_flight_log.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(df)} records")
    print(f"💾 Saved to: {output_path}")
    print(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return df

if __name__ == "__main__":
    generate_sample_flight_data()