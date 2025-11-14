import pandas as pd
from .log_parser_base import LogParser, TelemetryRecord
from datetime import datetime
from typing import List, Dict
import os

class CSVParser(LogParser):
    """Parser for CSV telemetry files"""
    
    def __init__(self, file_path: str, column_mapping: Dict[str, str] = None):
        super().__init__(file_path)
        self.column_mapping = column_mapping or {}
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse CSV file"""
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"📖 Parsing CSV: {self.file_path}")
        
        # Read CSV
        df = pd.read_csv(self.file_path)
        
        print(f"   Found columns: {list(df.columns)}")
        print(f"   Total rows: {len(df)}")
        
        self.records = []
        
        for idx, row in df.iterrows():
            record = TelemetryRecord(
                timestamp=self._parse_timestamp(row),
                latitude=self._get_value(row, 'latitude', 'lat', 'gps_lat'),
                longitude=self._get_value(row, 'longitude', 'lon', 'lng', 'gps_lon'),
                altitude=self._get_value(row, 'altitude', 'alt', 'height'),
                ground_speed=self._get_value(row, 'ground_speed', 'speed', 'velocity'),
                vertical_speed=self._get_value(row, 'vertical_speed', 'climb_rate', 'vz'),
                pitch=self._get_value(row, 'pitch'),
                roll=self._get_value(row, 'roll'),
                yaw=self._get_value(row, 'yaw', 'heading'),
                accel_x=self._get_value(row, 'accel_x', 'acc_x', 'ax'),
                accel_y=self._get_value(row, 'accel_y', 'acc_y', 'ay'),
                accel_z=self._get_value(row, 'accel_z', 'acc_z', 'az'),
                gyro_x=self._get_value(row, 'gyro_x', 'gyr_x', 'gx'),
                gyro_y=self._get_value(row, 'gyro_y', 'gyr_y', 'gy'),
                gyro_z=self._get_value(row, 'gyro_z', 'gyr_z', 'gz'),
                battery_voltage=self._get_value(row, 'battery_voltage', 'voltage', 'vbat', 'volt'),
                battery_current=self._get_value(row, 'battery_current', 'current', 'ibat', 'curr'),
                battery_remaining=self._get_value(row, 'battery_remaining', 'remaining', 'battery_pct'),
                battery_temperature=self._get_value(row, 'battery_temperature', 'battery_temp', 'bat_temp'),
                motor_1_rpm=self._get_value(row, 'motor_1_rpm', 'motor1_rpm', 'm1_rpm'),
                motor_2_rpm=self._get_value(row, 'motor_2_rpm', 'motor2_rpm', 'm2_rpm'),
                motor_3_rpm=self._get_value(row, 'motor_3_rpm', 'motor3_rpm', 'm3_rpm'),
                motor_4_rpm=self._get_value(row, 'motor_4_rpm', 'motor4_rpm', 'm4_rpm'),
                motor_5_rpm=self._get_value(row, 'motor_5_rpm', 'motor5_rpm', 'm5_rpm'),
                motor_6_rpm=self._get_value(row, 'motor_6_rpm', 'motor6_rpm', 'm6_rpm'),
                motor_1_current=self._get_value(row, 'motor_1_current', 'motor1_current', 'm1_current'),
                motor_2_current=self._get_value(row, 'motor_2_current', 'motor2_current', 'm2_current'),
                motor_3_current=self._get_value(row, 'motor_3_current', 'motor3_current', 'm3_current'),
                motor_4_current=self._get_value(row, 'motor_4_current', 'motor4_current', 'm4_current'),
                motor_5_current=self._get_value(row, 'motor_5_current', 'motor5_current', 'm5_current'),
                motor_6_current=self._get_value(row, 'motor_6_current', 'motor6_current', 'm6_current'),
                motor_1_temp=self._get_value(row, 'motor_1_temp', 'motor1_temp', 'm1_temp'),
                motor_2_temp=self._get_value(row, 'motor_2_temp', 'motor2_temp', 'm2_temp'),
                motor_3_temp=self._get_value(row, 'motor_3_temp', 'motor3_temp', 'm3_temp'),
                motor_4_temp=self._get_value(row, 'motor_4_temp', 'motor4_temp', 'm4_temp'),
                motor_5_temp=self._get_value(row, 'motor_5_temp', 'motor5_temp', 'm5_temp'),
                motor_6_temp=self._get_value(row, 'motor_6_temp', 'motor6_temp', 'm6_temp'),
                air_temperature=self._get_value(row, 'air_temperature', 'temp', 'temperature'),
                air_pressure=self._get_value(row, 'air_pressure', 'pressure', 'baro'),
                vibration_x=self._get_value(row, 'vibration_x', 'vib_x'),
                vibration_y=self._get_value(row, 'vibration_y', 'vib_y'),
                vibration_z=self._get_value(row, 'vibration_z', 'vib_z'),
                flight_mode=self._get_value(row, 'flight_mode', 'mode'),
                raw_data=row.to_dict()
            )
            
            self.records.append(record)
            
            if (idx + 1) % 1000 == 0:
                print(f"   Processed {idx + 1} rows...")
        
        print(f"✅ Parsed {len(self.records)} telemetry records")
        return self.records
    
    def _parse_timestamp(self, row) -> datetime:
        """Parse timestamp from row"""
        for col in ['timestamp', 'time', 'datetime', 'date', 'time_utc']:
            if col in row and pd.notna(row[col]):
                try:
                    return pd.to_datetime(row[col])
                except:
                    pass
        return datetime.now()
    
    def _get_value(self, row, *possible_names):
        """Try to get value from row using multiple possible column names"""
        for name in possible_names:
            if name in row and pd.notna(row[name]):
                try:
                    return float(row[name])
                except:
                    return row[name]
        return None
