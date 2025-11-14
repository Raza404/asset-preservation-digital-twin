import pandas as pd
from .log_parser_base import LogParser, TelemetryRecord
from datetime import datetime, timedelta
from typing import List
import os

class BetaflightParser(LogParser):
    """Parser for Betaflight blackbox logs"""
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse Betaflight CSV file"""
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"📖 Parsing Betaflight log: {self.file_path}")
        
        # Skip header rows (Betaflight has metadata at top)
        # The actual data starts after ~150 rows of metadata
        df = pd.read_csv(self.file_path, skiprows=151)
        
        print(f"   Found {len(df)} records with {len(df.columns)} columns")
        
        self.records = []
        
        # Start timestamp
        start_time = datetime.now()
        
        for idx, row in df.iterrows():
            # Calculate timestamp from time column (microseconds)
            if 'time' in row and pd.notna(row['time']):
                time_offset = row['time'] / 1_000_000  # Convert µs to seconds
                timestamp = start_time + timedelta(seconds=time_offset)
            else:
                timestamp = start_time + timedelta(seconds=idx * 0.001)
            
            # Extract gyro data (deg/s) - these are in deg/s * 1000 in Betaflight
            gyro_x = row.get('gyroADC[0]', 0) / 1000.0 if pd.notna(row.get('gyroADC[0]')) else None
            gyro_y = row.get('gyroADC[1]', 0) / 1000.0 if pd.notna(row.get('gyroADC[1]')) else None
            gyro_z = row.get('gyroADC[2]', 0) / 1000.0 if pd.notna(row.get('gyroADC[2]')) else None
            
            # Extract accelerometer data (in G * 1000, convert to m/s²)
            acc_x_raw = row.get('accSmooth[0]')
            acc_y_raw = row.get('accSmooth[1]')
            acc_z_raw = row.get('accSmooth[2]')
            
            # Convert from raw to m/s² (accSmooth is in units where 1000 = 1G)
            accel_x = (acc_x_raw / 1000.0) * 9.81 if pd.notna(acc_x_raw) else None
            accel_y = (acc_y_raw / 1000.0) * 9.81 if pd.notna(acc_y_raw) else None
            accel_z = (acc_z_raw / 1000.0) * 9.81 if pd.notna(acc_z_raw) else None
            
            # Extract heading (attitude) - already in radians
            roll = row.get('heading[0]')  # radians
            pitch = row.get('heading[1]')  # radians
            yaw = row.get('heading[2]')  # radians
            
            # Convert radians to degrees
            import math
            roll_deg = math.degrees(roll) if pd.notna(roll) else None
            pitch_deg = math.degrees(pitch) if pd.notna(pitch) else None
            yaw_deg = math.degrees(yaw) if pd.notna(yaw) else None
            
            # Battery (vbatLatest is in centivolts, so /100)
            battery_voltage = row.get('vbatLatest', 0) / 100.0 if pd.notna(row.get('vbatLatest')) else None
            
            # Current (amperageLatest is in centiamps, so /100)
            battery_current = row.get('amperageLatest', 0) / 100.0 if pd.notna(row.get('amperageLatest')) else None
            
            # GPS coordinates (in degrees * 10^7)
            gps_lat = row.get('GPS_coord[0]')
            gps_lon = row.get('GPS_coord[1]')
            
            if pd.notna(gps_lat) and gps_lat != 0:
                latitude = gps_lat / 10_000_000.0
            else:
                latitude = None
            
            if pd.notna(gps_lon) and gps_lon != 0:
                longitude = gps_lon / 10_000_000.0
            else:
                longitude = None
            
            # GPS altitude (centimeters to meters)
            gps_alt = row.get('GPS_altitude')
            altitude = gps_alt / 100.0 if pd.notna(gps_alt) and gps_alt != 0 else None
            
            # GPS speed (cm/s to m/s)
            gps_speed = row.get('GPS_speed')
            ground_speed = gps_speed / 100.0 if pd.notna(gps_speed) else None
            
            # Barometer altitude (centimeters to meters)
            baro_alt = row.get('baroAlt')
            baro_altitude = baro_alt / 100.0 if pd.notna(baro_alt) else None
            
            # Use barometer altitude if GPS not available
            if altitude is None and baro_altitude is not None:
                altitude = baro_altitude
            
            # Motor values (0-2000 scale)
            motor_values = []
            for i in range(4):
                motor = row.get(f'motor[{i}]')
                if pd.notna(motor):
                    motor_values.append(int(motor))
            
            # eRPM values
            erpm_values = []
            for i in range(4):
                erpm = row.get(f'eRPM[{i}]')
                if pd.notna(erpm):
                    erpm_values.append(int(erpm))
            
            # Create record
            record = TelemetryRecord(
                timestamp=timestamp,
                
                # Position
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                
                # Velocity
                ground_speed=ground_speed,
                
                # IMU - Accelerometer (converted to m/s²)
                accel_x=accel_x,
                accel_y=accel_y,
                accel_z=accel_z,
                
                # IMU - Gyroscope (deg/s)
                gyro_x=gyro_x,
                gyro_y=gyro_y,
                gyro_z=gyro_z,
                
                # Attitude (degrees)
                roll=roll_deg,
                pitch=pitch_deg,
                yaw=yaw_deg,
                
                # Battery
                battery_voltage=battery_voltage,
                battery_current=battery_current,
                
                # Store raw data
                raw_data={
                    'motors': motor_values,
                    'erpm': erpm_values,
                    'loop_iteration': int(row.get('loopIteration', 0)),
                    'rssi': int(row.get('rssi', 0)) if pd.notna(row.get('rssi')) else None,
                    'gps_num_sat': int(row.get('GPS_numSat', 0)) if pd.notna(row.get('GPS_numSat')) else None,
                    'source': 'betaflight'
                }
            )
            
            self.records.append(record)
            
            if (idx + 1) % 50000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} records...")
        
        print(f"✅ Parsed {len(self.records)} telemetry records")
        return self.records