import pandas as pd
from .log_parser_base import LogParser, TelemetryRecord
from datetime import datetime
from typing import List
import os

class UAVNavigationParser(LogParser):
    """Parser for UAV Autonomous Navigation dataset"""
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse UAV Navigation CSV file"""
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"📖 Parsing UAV Navigation: {self.file_path}")
        
        # Read CSV
        df = pd.read_csv(self.file_path)
        
        print(f"   Found {len(df)} records with {len(df.columns)} columns")
        
        self.records = []
        
        for idx, row in df.iterrows():
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(row['timestamp'])
            except:
                timestamp = datetime.now()
            
            # Estimate voltage from battery percentage (4S LiPo)
            battery_pct = row.get('battery_level')
            if pd.notna(battery_pct):
                battery_voltage = 12.8 + (battery_pct / 100.0) * 4.0  # 16.8V to 12.8V
            else:
                battery_voltage = None
            
            # Create telemetry record
            record = TelemetryRecord(
                timestamp=timestamp,
                
                # Position (excellent GPS data!)
                latitude=row.get('latitude'),
                longitude=row.get('longitude'),
                altitude=row.get('altitude'),
                
                # Velocity
                ground_speed=row.get('speed'),
                
                # IMU - Accelerometer (PERFECT!)
                accel_x=row.get('imu_acc_x'),
                accel_y=row.get('imu_acc_y'),
                accel_z=row.get('imu_acc_z'),
                
                # IMU - Gyroscope (PERFECT!)
                gyro_x=row.get('imu_gyro_x'),
                gyro_y=row.get('imu_gyro_y'),
                gyro_z=row.get('imu_gyro_z'),
                
                # Battery
                battery_remaining=battery_pct,
                battery_voltage=battery_voltage,
                
                # Environmental
                wind_speed=row.get('wind_speed'),
                
                # Store additional data
                raw_data={
                    'lidar_distance': row.get('lidar_distance'),
                    'obstacle_detected': row.get('obstacle_detected')
                }
            )
            
            self.records.append(record)
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} records...")
        
        print(f"✅ Parsed {len(self.records)} telemetry records")
        return self.records