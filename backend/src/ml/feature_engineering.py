import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from data_ingestion.log_parser_base import TelemetryRecord

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    
    # Time-based features
    window_size_seconds: float = 1.0
    rolling_window_size: int = 10
    
    # Statistical features
    include_statistical: bool = True
    include_frequency: bool = True
    include_derivative: bool = True
    
    # Domain-specific features
    include_flight_dynamics: bool = True
    include_energy_metrics: bool = True
    include_stress_indicators: bool = True
    
    # Handling missing data
    fill_method: str = 'interpolate'  # 'interpolate', 'forward', 'zero'


class UniversalFeatureExtractor:
    """
    Universal feature extractor that works with ANY drone telemetry data.
    Automatically adapts to available sensors and data quality.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.available_sensors = set()
        self.feature_names = []
    
    def extract_features(self, records: List[TelemetryRecord]) -> pd.DataFrame:
        """
        Extract features from telemetry records.
        Works with ANY combination of available sensors.
        """
        
       # print(f"🔬 Extracting features from {len(records)} records...")
        
        # Convert to DataFrame
        df = self._records_to_dataframe(records)
        
        # Detect available sensors
        self._detect_available_sensors(df)
       # print(f"   Detected sensors: {', '.join(sorted(self.available_sensors))}")
        
        # Extract features based on available data
        features = pd.DataFrame()
        
        # Time-based features (always available)
        features = pd.concat([features, self._extract_time_features(df)], axis=1)
        
        # IMU features (if accelerometer/gyro available)
        if 'imu' in self.available_sensors:
            features = pd.concat([features, self._extract_imu_features(df)], axis=1)
        
        # GPS features (if GPS available)
        if 'gps' in self.available_sensors:
            features = pd.concat([features, self._extract_gps_features(df)], axis=1)
        
        # Attitude features (if attitude data available)
        if 'attitude' in self.available_sensors:
            features = pd.concat([features, self._extract_attitude_features(df)], axis=1)
        
        # Battery features (if battery data available)
        if 'battery' in self.available_sensors:
            features = pd.concat([features, self._extract_battery_features(df)], axis=1)
        
        # Motion features (derived from available data)
        features = pd.concat([features, self._extract_motion_features(df)], axis=1)
        
        # Statistical features (rolling windows)
        if self.config.include_statistical:
            features = pd.concat([features, self._extract_statistical_features(features)], axis=1)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        self.feature_names = features.columns.tolist()
        #print(f"   ✅ Extracted {len(self.feature_names)} features")
        
        return features
    
    def _records_to_dataframe(self, records: List[TelemetryRecord]) -> pd.DataFrame:
        """Convert records to DataFrame"""
        
        data = []
        for r in records:
            data.append({
                'timestamp': r.timestamp,
                'accel_x': r.accel_x,
                'accel_y': r.accel_y,
                'accel_z': r.accel_z,
                'gyro_x': r.gyro_x,
                'gyro_y': r.gyro_y,
                'gyro_z': r.gyro_z,
                'latitude': r.latitude,
                'longitude': r.longitude,
                'altitude': r.altitude,
                'ground_speed': r.ground_speed,
                'vertical_speed': r.vertical_speed,
                'roll': r.roll,
                'pitch': r.pitch,
                'yaw': r.yaw,
                'battery_voltage': r.battery_voltage,
                'battery_current': r.battery_current,
                'battery_remaining': r.battery_remaining,
                'wind_speed': r.wind_speed,
                'air_temperature': r.air_temperature
            })
        
        df = pd.DataFrame(data)
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds()
        
        return df
    
    def _detect_available_sensors(self, df: pd.DataFrame):
        """Detect which sensors have valid data"""
        
        self.available_sensors = set()
        
        # IMU (accelerometer + gyro)
        if df[['accel_x', 'accel_y', 'accel_z']].notna().sum().sum() > len(df) * 0.1:
            self.available_sensors.add('imu')
        
        # GPS
        if df[['latitude', 'longitude']].notna().sum().sum() > len(df) * 0.1:
            self.available_sensors.add('gps')
        
        # Attitude
        if df[['roll', 'pitch', 'yaw']].notna().sum().sum() > len(df) * 0.1:
            self.available_sensors.add('attitude')
        
        # Battery
        if df['battery_voltage'].notna().sum() > len(df) * 0.1:
            self.available_sensors.add('battery')
        
        # Altitude
        if df['altitude'].notna().sum() > len(df) * 0.1:
            self.available_sensors.add('altitude')
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features (always available)"""
        
        features = pd.DataFrame(index=df.index)
        
        # Sample rate
        features['sample_rate_hz'] = 1.0 / df['time_delta'].replace(0, np.nan)
        
        # Time since start
        if len(df) > 0:
            start_time = df['timestamp'].iloc[0]
            features['time_since_start'] = (df['timestamp'] - start_time).dt.total_seconds()
        
        # Sequence index
        features['sequence_index'] = np.arange(len(df))
        
        return features
    
    def _extract_imu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract IMU-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # G-force magnitude
        accel_mag = np.sqrt(
            df['accel_x'].fillna(0)**2 + 
            df['accel_y'].fillna(0)**2 + 
            df['accel_z'].fillna(0)**2
        )
        features['g_force'] = accel_mag / 9.81
        
        # Individual axis G-forces
        features['g_force_x'] = df['accel_x'].fillna(0) / 9.81
        features['g_force_y'] = df['accel_y'].fillna(0) / 9.81
        features['g_force_z'] = df['accel_z'].fillna(0) / 9.81
        
        # Gyro magnitude (rotation rate)
        if df[['gyro_x', 'gyro_y', 'gyro_z']].notna().any().any():
            gyro_mag = np.sqrt(
                df['gyro_x'].fillna(0)**2 + 
                df['gyro_y'].fillna(0)**2 + 
                df['gyro_z'].fillna(0)**2
            )
            features['rotation_rate'] = gyro_mag
            features['gyro_x_rate'] = df['gyro_x'].fillna(0)
            features['gyro_y_rate'] = df['gyro_y'].fillna(0)
            features['gyro_z_rate'] = df['gyro_z'].fillna(0)
        
        # Jerk (rate of change of acceleration)
        if self.config.include_derivative:
            features['jerk_x'] = df['accel_x'].diff() / df['time_delta']
            features['jerk_y'] = df['accel_y'].diff() / df['time_delta']
            features['jerk_z'] = df['accel_z'].diff() / df['time_delta']
            features['jerk_magnitude'] = np.sqrt(
                features['jerk_x'].fillna(0)**2 + 
                features['jerk_y'].fillna(0)**2 + 
                features['jerk_z'].fillna(0)**2
            )
        
        # Vibration indicator (high-frequency acceleration changes)
        if len(df) > 10:
            features['vibration_x'] = df['accel_x'].rolling(window=10, min_periods=1).std()
            features['vibration_y'] = df['accel_y'].rolling(window=10, min_periods=1).std()
            features['vibration_z'] = df['accel_z'].rolling(window=10, min_periods=1).std()
        
        return features
    
    def _extract_gps_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract GPS-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Speed
        if df['ground_speed'].notna().any():
            features['ground_speed'] = df['ground_speed'].fillna(0)
            features['speed_change_rate'] = df['ground_speed'].diff() / df['time_delta']
        
        # Altitude
        if df['altitude'].notna().any():
            features['altitude'] = df['altitude'].fillna(0)
            features['altitude_rate'] = df['altitude'].diff() / df['time_delta']
        
        # 3D speed (if we have altitude rate)
        if 'altitude_rate' in features.columns and 'ground_speed' in features.columns:
            features['speed_3d'] = np.sqrt(
                features['ground_speed']**2 + features['altitude_rate']**2
            )
        
        # Distance traveled (cumulative)
        if df[['latitude', 'longitude']].notna().all(axis=1).any():
            # Simplified distance (works for short distances)
            lat_diff = df['latitude'].diff().fillna(0)
            lon_diff = df['longitude'].diff().fillna(0)
            features['distance_delta'] = np.sqrt(lat_diff**2 + lon_diff**2) * 111000  # rough m conversion
            features['distance_traveled'] = features['distance_delta'].cumsum()
        
        return features
    
    def _extract_attitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract attitude-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Attitude angles
        features['roll'] = df['roll'].fillna(0)
        features['pitch'] = df['pitch'].fillna(0)
        features['yaw'] = df['yaw'].fillna(0)
        
        # Attitude rates (angular velocity)
        if self.config.include_derivative:
            features['roll_rate'] = df['roll'].diff() / df['time_delta']
            features['pitch_rate'] = df['pitch'].diff() / df['time_delta']
            features['yaw_rate'] = df['yaw'].diff() / df['time_delta']
        
        # Attitude magnitude (how tilted)
        features['tilt_angle'] = np.sqrt(df['roll'].fillna(0)**2 + df['pitch'].fillna(0)**2)
        
        return features
    
    def _extract_battery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract battery-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Voltage
        if df['battery_voltage'].notna().any():
            features['battery_voltage'] = df['battery_voltage'].fillna(method='ffill')
            features['voltage_drop_rate'] = df['battery_voltage'].diff() / df['time_delta']
        
        # Current
        if df['battery_current'].notna().any():
            features['battery_current'] = df['battery_current'].fillna(0)
            
            # Power (voltage * current)
            if 'battery_voltage' in features.columns:
                features['power_watts'] = features['battery_voltage'] * features['battery_current']
        
        # Remaining percentage
        if df['battery_remaining'].notna().any():
            features['battery_remaining_pct'] = df['battery_remaining'].fillna(method='ffill')
        
        return features
    
    def _extract_motion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract motion-related features from available data"""
        
        features = pd.DataFrame(index=df.index)
        
        # Energy-based features (if we have mass and velocity)
        if df['ground_speed'].notna().any():
            # Kinetic energy indicator (proportional, without mass)
            features['kinetic_energy_indicator'] = df['ground_speed'].fillna(0)**2
        
        # Maneuver intensity (combination of rotation and acceleration)
        if 'imu' in self.available_sensors:
            g_force = np.sqrt(
                df['accel_x'].fillna(0)**2 + 
                df['accel_y'].fillna(0)**2 + 
                df['accel_z'].fillna(0)**2
            ) / 9.81
            
            if df[['gyro_x', 'gyro_y', 'gyro_z']].notna().any().any():
                rotation = np.sqrt(
                    df['gyro_x'].fillna(0)**2 + 
                    df['gyro_y'].fillna(0)**2 + 
                    df['gyro_z'].fillna(0)**2
                )
                features['maneuver_intensity'] = g_force * rotation
            else:
                features['maneuver_intensity'] = g_force
        
        return features
    
    def _extract_statistical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features using rolling windows"""
        
        stat_features = pd.DataFrame(index=features.index)
        
        window = self.config.rolling_window_size
        
        # Select numeric columns for statistical analysis
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        key_cols = [col for col in numeric_cols if any(
            keyword in col.lower() for keyword in 
            ['g_force', 'rotation', 'speed', 'altitude', 'vibration', 'jerk']
        )]
        
        for col in key_cols[:10]:  # Limit to avoid feature explosion
            if col in features.columns:
                # Rolling mean
                stat_features[f'{col}_rolling_mean'] = features[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std (volatility)
                stat_features[f'{col}_rolling_std'] = features[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling max
                stat_features[f'{col}_rolling_max'] = features[col].rolling(
                    window=window, min_periods=1
                ).max()
        
        return stat_features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        
        if self.config.fill_method == 'interpolate':
            features = features.interpolate(method='linear', limit_direction='both')
        elif self.config.fill_method == 'forward':
            features = features.fillna(method='ffill').fillna(method='bfill')
        elif self.config.fill_method == 'zero':
            features = features.fillna(0)
        
        # Fill any remaining NaNs with 0
        features = features.fillna(0)
        
        # Replace inf values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of extracted feature names"""
        return self.feature_names