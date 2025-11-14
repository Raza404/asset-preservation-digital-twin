import pandas as pd
from .log_parser_base import LogParser, TelemetryRecord
from datetime import datetime
from typing import List
import os

class SurveilDroneParser(LogParser):
    """Parser for SurveilDrone-Net23 dataset"""
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse SurveilDrone-Net23 CSV file"""
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"📖 Parsing SurveilDrone-Net23: {self.file_path}")
        
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
            
            # Calculate ground speed from velocity components
            try:
                import math
                if pd.notna(row['velocity_x']) and pd.notna(row['velocity_y']):
                    ground_speed = math.sqrt(row['velocity_x']**2 + row['velocity_y']**2)
                else:
                    ground_speed = None
            except:
                ground_speed = None
            
            # Estimate voltage from battery percentage (4S LiPo)
            battery_pct = row.get('battery_level_pct')
            if pd.notna(battery_pct):
                battery_voltage = 12.8 + (battery_pct / 100.0) * 4.0
            else:
                battery_voltage = None
            
            # Create telemetry record
            record = TelemetryRecord(
                timestamp=timestamp,
                
                # Position
                latitude=row.get('gps_lat'),
                longitude=row.get('gps_lon'),
                altitude=row.get('altitude_m'),
                
                # Velocity
                ground_speed=ground_speed,
                vertical_speed=row.get('velocity_z'),
                
                # Attitude (only have heading/yaw)
                yaw=row.get('heading_deg'),
                
                # IMU - Accelerometer
                accel_x=row.get('acceleration_x'),
                accel_y=row.get('acceleration_y'),
                accel_z=row.get('acceleration_z'),
                
                # Battery
                battery_remaining=battery_pct,
                battery_voltage=battery_voltage,
                
                # Environmental
                air_temperature=row.get('ambient_temp_C'),
                wind_speed=row.get('wind_speed_mps'),
                wind_direction=row.get('wind_dir_deg'),
                
                # Store raw data
                raw_data={
                    'drone_id': row.get('drone_id'),
                    'mission_id': row.get('mission_id'),
                    'mission_type': row.get('mission_type'),
                    'power_consumption_watts': row.get('power_consumption_watts'),
                    'flight_time_s': row.get('flight_time_s'),
                }
            )
            
            self.records.append(record)
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} records...")
        
        print(f"✅ Parsed {len(self.records)} telemetry records")
        return self.records