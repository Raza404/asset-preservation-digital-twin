from pymavlink import mavutil
from .log_parser_base import LogParser, TelemetryRecord
from datetime import datetime
from typing import List
import os

class ArduPilotParser(LogParser):
    """Parser for ArduPilot .bin log files"""
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse ArduPilot binary log file"""
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Log file not found: {self.file_path}")
        
        print(f"📖 Parsing ArduPilot log: {self.file_path}")
        
        try:
            mlog = mavutil.mavlink_connection(self.file_path)
        except Exception as e:
            print(f"❌ Error opening log file: {e}")
            return []
        
        messages_by_time = {}
        
        while True:
            msg = mlog.recv_match(blocking=False)
            if msg is None:
                break
            
            msg_type = msg.get_type()
            timestamp = self._get_timestamp(msg)
            
            if timestamp not in messages_by_time:
                messages_by_time[timestamp] = {}
            
            messages_by_time[timestamp][msg_type] = msg
        
        print(f"   Found {len(messages_by_time)} unique timestamps")
        
        self.records = []
        for timestamp, messages in sorted(messages_by_time.items()):
            record = self._create_record(timestamp, messages)
            if record:
                self.records.append(record)
        
        print(f"✅ Parsed {len(self.records)} telemetry records")
        return self.records
    
    def _create_record(self, timestamp: datetime, messages: dict) -> TelemetryRecord:
        """Create telemetry record from messages at this timestamp"""
        
        record = TelemetryRecord(timestamp=timestamp)
        
        # GPS data
        if 'GPS' in messages:
            gps = messages['GPS']
            record.latitude = getattr(gps, 'Lat', None)
            record.longitude = getattr(gps, 'Lng', None)
            record.altitude = getattr(gps, 'Alt', None)
            record.ground_speed = getattr(gps, 'Spd', None)
        
        # IMU data
        if 'IMU' in messages:
            imu = messages['IMU']
            record.accel_x = getattr(imu, 'AccX', None)
            record.accel_y = getattr(imu, 'AccY', None)
            record.accel_z = getattr(imu, 'AccZ', None)
            record.gyro_x = getattr(imu, 'GyrX', None)
            record.gyro_y = getattr(imu, 'GyrY', None)
            record.gyro_z = getattr(imu, 'GyrZ', None)
        
        # Battery data
        if 'BATT' in messages or 'BAT' in messages:
            batt = messages.get('BATT') or messages.get('BAT')
            record.battery_voltage = getattr(batt, 'Volt', None)
            record.battery_current = getattr(batt, 'Curr', None)
            record.battery_remaining = getattr(batt, 'CurrTot', None)
        
        # Attitude data
        if 'ATT' in messages:
            att = messages['ATT']
            record.pitch = getattr(att, 'Pitch', None)
            record.roll = getattr(att, 'Roll', None)
            record.yaw = getattr(att, 'Yaw', None)
        
        # Vibration
        if 'VIBE' in messages:
            vibe = messages['VIBE']
            record.vibration_x = getattr(vibe, 'VibeX', None)
            record.vibration_y = getattr(vibe, 'VibeY', None)
            record.vibration_z = getattr(vibe, 'VibeZ', None)
        
        return record
    
    def _get_timestamp(self, msg) -> datetime:
        """Extract timestamp from message"""
        if hasattr(msg, 'TimeUS'):
            return datetime.fromtimestamp(msg.TimeUS / 1e6)
        elif hasattr(msg, '_timestamp'):
            return datetime.fromtimestamp(msg._timestamp)
        else:
            return datetime.now()
