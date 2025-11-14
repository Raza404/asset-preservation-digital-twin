"""
Anomaly Detection Module for Drone Failure Prediction
Uses ML-based approach to detect anomalies in drone telemetry
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """ML-based anomaly detector for drone telemetry data"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of base estimators in the ensemble
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'battery_level', 'temperature', 'vibration', 
            'altitude', 'speed', 'motor_current'
        ]
    
    def train(self, telemetry_data: np.ndarray) -> None:
        """
        Train the anomaly detection model
        
        Args:
            telemetry_data: Historical telemetry data for training
        """
        if telemetry_data.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {telemetry_data.shape[1]}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(telemetry_data)
        
        # Train the model
        self.model.fit(scaled_data)
        self.is_trained = True
    
    def predict(self, telemetry_sample: np.ndarray) -> Tuple[int, float]:
        """
        Predict if a telemetry sample is anomalous
        
        Args:
            telemetry_sample: Single telemetry reading
            
        Returns:
            Tuple of (prediction, anomaly_score) where:
                prediction: 1 for normal, -1 for anomaly
                anomaly_score: Normalized anomaly score (0-1, higher means more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Reshape if necessary
        if telemetry_sample.ndim == 1:
            telemetry_sample = telemetry_sample.reshape(1, -1)
        
        # Scale the data
        scaled_sample = self.scaler.transform(telemetry_sample)
        
        # Get prediction and anomaly score
        prediction = self.model.predict(scaled_sample)[0]
        anomaly_score = -self.model.score_samples(scaled_sample)[0]
        
        # Normalize anomaly score to 0-1 range
        anomaly_score = 1 / (1 + np.exp(-anomaly_score))
        
        return prediction, anomaly_score
    
    def detect_failure_risk(self, telemetry_sample: np.ndarray) -> Dict[str, any]:
        """
        Detect failure risk and provide detailed diagnostics
        
        Args:
            telemetry_sample: Single telemetry reading
            
        Returns:
            Dictionary with failure risk assessment
        """
        prediction, anomaly_score = self.predict(telemetry_sample)
        
        # Determine risk level
        if anomaly_score < 0.3:
            risk_level = "LOW"
        elif anomaly_score < 0.6:
            risk_level = "MEDIUM"
        elif anomaly_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(anomaly_score),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(risk_level, telemetry_sample)
        }
    
    def _get_recommendation(self, risk_level: str, telemetry_sample: np.ndarray) -> str:
        """Generate recommendation based on risk level and telemetry"""
        if risk_level == "CRITICAL":
            return "IMMEDIATE ACTION: Land drone immediately and perform maintenance"
        elif risk_level == "HIGH":
            return "Return to base for inspection"
        elif risk_level == "MEDIUM":
            return "Monitor closely and reduce flight intensity"
        else:
            return "Continue normal operations"
