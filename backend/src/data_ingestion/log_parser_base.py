from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class TelemetryRecord:
    """Standardized telemetry data structure"""
    timestamp: datetime
    
    # Position
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    
    # Velocity
    ground_speed: Optional[float] = None
    vertical_speed: Optional[float] = None
    
    # Attitude
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yaw: Optional[float] = None
    
    # IMU
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    
    # Battery
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    battery_remaining: Optional[float] = None
    battery_temperature: Optional[float] = None
    
    # Motors (4 motors)
    motor_1_rpm: Optional[float] = None
    motor_2_rpm: Optional[float] = None
    motor_3_rpm: Optional[float] = None
    motor_4_rpm: Optional[float] = None
    motor_1_current: Optional[float] = None
    motor_2_current: Optional[float] = None
    motor_3_current: Optional[float] = None
    motor_4_current: Optional[float] = None
    motor_1_temp: Optional[float] = None
    motor_2_temp: Optional[float] = None
    motor_3_temp: Optional[float] = None
    motor_4_temp: Optional[float] = None
    
    # Motors 5-6 (for hexacopters)
    motor_5_rpm: Optional[float] = None
    motor_6_rpm: Optional[float] = None
    motor_5_current: Optional[float] = None
    motor_6_current: Optional[float] = None
    motor_5_temp: Optional[float] = None
    motor_6_temp: Optional[float] = None
    
    # Environmental
    air_temperature: Optional[float] = None
    air_pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    
    # Vibration
    vibration_x: Optional[float] = None
    vibration_y: Optional[float] = None
    vibration_z: Optional[float] = None
    
    # Flight mode
    flight_mode: Optional[str] = None
    armed: Optional[bool] = None
    
    # Raw data
    raw_data: Optional[Dict[str, Any]] = None

class LogParser(ABC):
    """Base class for all log parsers"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.records: List[TelemetryRecord] = []
    
    @abstractmethod
    def parse(self) -> List[TelemetryRecord]:
        """Parse the log file and return telemetry records"""
        pass
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert parsed records to pandas DataFrame"""
        if not self.records:
            self.parse()
        
        return pd.DataFrame([asdict(record) for record in self.records])
    
    def save_to_csv(self, output_path: str):
        """Save parsed data to CSV"""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} records to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the flight"""
        if not self.records:
            self.parse()
        
        df = self.to_dataframe()
        
        return {
            "total_records": len(df),
            "duration_seconds": (df['timestamp'].max() - df['timestamp'].min()).total_seconds() if len(df) > 0 else 0,
            "max_altitude": df['altitude'].max() if 'altitude' in df else None,
            "max_speed": df['ground_speed'].max() if 'ground_speed' in df else None,
    }
