"""
Drone Telemetry Simulator for Testing
Generates realistic drone telemetry data with anomalies
"""

import numpy as np
from typing import Tuple, Dict


class TelemetrySimulator:
    """Simulates realistic drone telemetry data"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the telemetry simulator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Normal operating ranges
        self.normal_ranges = {
            'battery_level': (80, 100),
            'temperature': (20, 35),
            'vibration': (0, 2),
            'altitude': (50, 100),
            'speed': (5, 15),
            'motor_current': (2, 5)
        }
        
        # Anomaly ranges
        self.anomaly_ranges = {
            'battery_level': (10, 40),
            'temperature': (45, 70),
            'vibration': (5, 15),
            'altitude': (5, 30),
            'speed': (0.5, 3),
            'motor_current': (8, 15)
        }
    
    def generate_normal_telemetry(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate normal telemetry data
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of telemetry samples
        """
        samples = []
        
        for _ in range(num_samples):
            sample = [
                np.random.uniform(*self.normal_ranges['battery_level']),
                np.random.uniform(*self.normal_ranges['temperature']),
                np.random.uniform(*self.normal_ranges['vibration']),
                np.random.uniform(*self.normal_ranges['altitude']),
                np.random.uniform(*self.normal_ranges['speed']),
                np.random.uniform(*self.normal_ranges['motor_current'])
            ]
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_anomalous_telemetry(self, num_samples: int = 1,
                                    anomaly_type: str = 'random') -> np.ndarray:
        """
        Generate anomalous telemetry data
        
        Args:
            num_samples: Number of samples to generate
            anomaly_type: Type of anomaly ('battery', 'temperature', 'vibration', 'random')
            
        Returns:
            Array of anomalous telemetry samples
        """
        samples = []
        
        for _ in range(num_samples):
            # Start with normal values
            sample = [
                np.random.uniform(*self.normal_ranges['battery_level']),
                np.random.uniform(*self.normal_ranges['temperature']),
                np.random.uniform(*self.normal_ranges['vibration']),
                np.random.uniform(*self.normal_ranges['altitude']),
                np.random.uniform(*self.normal_ranges['speed']),
                np.random.uniform(*self.normal_ranges['motor_current'])
            ]
            
            # Introduce anomalies based on type
            if anomaly_type == 'battery' or (anomaly_type == 'random' and np.random.random() < 0.3):
                sample[0] = np.random.uniform(*self.anomaly_ranges['battery_level'])
            
            if anomaly_type == 'temperature' or (anomaly_type == 'random' and np.random.random() < 0.3):
                sample[1] = np.random.uniform(*self.anomaly_ranges['temperature'])
            
            if anomaly_type == 'vibration' or (anomaly_type == 'random' and np.random.random() < 0.3):
                sample[2] = np.random.uniform(*self.anomaly_ranges['vibration'])
                sample[5] = np.random.uniform(*self.anomaly_ranges['motor_current'])
            
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_training_dataset(self, num_samples: int = 1000,
                                 contamination: float = 0.1) -> np.ndarray:
        """
        Generate training dataset with normal and anomalous data
        
        Args:
            num_samples: Total number of samples
            contamination: Proportion of anomalous samples
            
        Returns:
            Mixed training dataset
        """
        num_anomalies = int(num_samples * contamination)
        num_normal = num_samples - num_anomalies
        
        normal_data = self.generate_normal_telemetry(num_normal)
        anomalous_data = self.generate_anomalous_telemetry(num_anomalies)
        
        # Combine and shuffle
        training_data = np.vstack([normal_data, anomalous_data])
        np.random.shuffle(training_data)
        
        return training_data
    
    def simulate_flight_sequence(self, duration: int = 20,
                                anomaly_start: int = 10) -> Tuple[np.ndarray, list]:
        """
        Simulate a complete flight sequence with telemetry
        
        Args:
            duration: Number of time steps
            anomaly_start: Time step when anomaly begins
            
        Returns:
            Tuple of (telemetry_sequence, positions)
        """
        telemetry_sequence = []
        positions = []
        
        # Simulate normal flight
        for i in range(duration):
            if i < anomaly_start:
                # Normal operation
                telemetry = self.generate_normal_telemetry(1)[0]
            else:
                # Gradual degradation
                severity = (i - anomaly_start) / (duration - anomaly_start)
                telemetry = self._generate_degrading_telemetry(severity)
            
            telemetry_sequence.append(telemetry)
            
            # Simulate position (simple trajectory)
            x = i * 5.0
            y = i * 3.0
            z = 50.0 + np.sin(i * 0.3) * 10.0
            positions.append((x, y, z))
        
        return np.array(telemetry_sequence), positions
    
    def _generate_degrading_telemetry(self, severity: float) -> np.ndarray:
        """Generate telemetry with degradation based on severity (0-1)"""
        # Normal baseline
        telemetry = self.generate_normal_telemetry(1)[0]
        
        # Apply degradation
        telemetry[0] -= severity * 50  # Battery draining
        telemetry[1] += severity * 30  # Temperature rising
        telemetry[2] += severity * 8   # Vibration increasing
        telemetry[5] += severity * 7   # Motor current increasing
        
        return telemetry
    
    def get_feature_names(self) -> list:
        """Get list of telemetry feature names"""
        return ['battery_level', 'temperature', 'vibration', 
                'altitude', 'speed', 'motor_current']
