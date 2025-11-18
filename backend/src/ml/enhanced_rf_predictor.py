"""
Enhanced Random Forest classifier with component health tracking and recommendations.
Provides fast (<50ms) anomaly detection with actionable insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path


class EnhancedRandomForestPredictor:
    """
    Enhanced Random Forest for real-time failure detection with component health analysis.
    
    Features:
    - Fast inference (<50ms)
    - Component-specific failure classification
    - Health degradation tracking
    - Actionable recommendations
    - Feature importance analysis
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize enhanced RF predictor.
        
        Args:
            model_path: Path to trained RandomForest model (.joblib)
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = []
        self.health_history = {}  # Track component health over time
        
        # Component feature mappings
        self.component_features = {
            'motor': ['rpm', 'motor_temp', 'motor_current', 'motor_vibration'],
            'battery': ['battery_voltage', 'battery_current', 'battery_remaining', 'power_draw'],
            'esc': ['esc_temp', 'throttle', 'current_draw', 'voltage_ripple'],
            'gyro': ['gyro_x', 'gyro_y', 'gyro_z', 'angular_velocity'],
            'accelerometer': ['accel_x', 'accel_y', 'accel_z', 'g_force'],
            'gps': ['gps_satellites', 'hdop', 'gps_fix_quality'],
            'propeller': ['vibration_magnitude', 'jerk', 'oscillation']
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str):
        """Load trained Random Forest model."""
        try:
            self.model = joblib.load(path)
            print(f"✓ Random Forest model loaded from {path}")
            
            # Extract feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
        except Exception as e:
            print(f"⚠️  Could not load Random Forest model: {e}")
    
    def predict_with_health_analysis(self, features: pd.DataFrame) -> Dict:
        """
        Predict failures and analyze component health.
        
        Args:
            features: DataFrame with telemetry features
        
        Returns:
            Comprehensive health analysis with recommendations
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get RF prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Map to failure type
        failure_type_map = {
            0: 'normal',
            1: 'motor_failure',
            2: 'vibration_anomaly'
        }
        
        predicted_type = failure_type_map.get(prediction, 'unknown')
        confidence = float(max(probabilities))
        
        # Analyze each component
        component_health = self._analyze_all_components(features, probabilities)
        
        # Generate overall assessment
        overall_risk = self._calculate_overall_risk(component_health)
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(
            component_health, overall_risk, predicted_type
        )
        
        return {
            'prediction': predicted_type,
            'confidence': confidence,
            'overall_risk': overall_risk,
            'component_health': component_health,
            'recommendations': recommendations,
            'feature_importance': self._get_top_features(features),
            'timestamp': pd.Timestamp.now()
        }
    
    def _analyze_all_components(self, 
                                 features: pd.DataFrame,
                                 failure_probabilities: np.ndarray) -> Dict[str, Dict]:
        """
        Analyze health of all drone components.
        
        Args:
            features: Telemetry features
            failure_probabilities: Model output probabilities
        
        Returns:
            Component health dictionary
        """
        component_health = {}
        
        # Extract feature values
        feature_dict = features.iloc[0].to_dict() if isinstance(features, pd.DataFrame) else features
        
        # Motor health
        component_health['motor'] = self._analyze_motor_health(
            feature_dict, failure_probabilities
        )
        
        # Battery health
        component_health['battery'] = self._analyze_battery_health(
            feature_dict, failure_probabilities
        )
        
        # ESC health
        component_health['esc'] = self._analyze_esc_health(
            feature_dict, failure_probabilities
        )
        
        # Sensor health (gyro, accel, GPS)
        component_health['sensors'] = self._analyze_sensor_health(
            feature_dict, failure_probabilities
        )
        
        # Propeller/vibration health
        component_health['propeller'] = self._analyze_propeller_health(
            feature_dict, failure_probabilities
        )
        
        return component_health
    
    def _analyze_motor_health(self, features: Dict, probs: np.ndarray) -> Dict:
        """Analyze motor health from features."""
        # Extract motor-related features
        motor_features = self._extract_component_features(features, 'motor')
        
        # Calculate health score (0-1, where 1 is healthy)
        health_score = 1.0 - probs[1] * 0.8  # Motor failure probability
        
        # Check for specific issues
        issues = []
        if any('motor_temp' in k and features.get(k, 0) > 80 for k in features):
            issues.append("High motor temperature detected")
            health_score *= 0.9
        
        if any('motor_current' in k and features.get(k, 0) > 30 for k in features):
            issues.append("Excessive current draw")
            health_score *= 0.85
        
        if any('motor_vibration' in k and features.get(k, 0) > 0.8 for k in features):
            issues.append("Abnormal vibration pattern")
            health_score *= 0.8
        
        status = 'CRITICAL' if health_score < 0.3 else 'WARNING' if health_score < 0.6 else 'HEALTHY'
        
        return {
            'health_score': float(health_score),
            'status': status,
            'issues': issues,
            'rul_estimate': self._estimate_rul(health_score, 'motor'),
            'degradation_rate': self._calculate_degradation_rate('motor', health_score)
        }
    
    def _analyze_battery_health(self, features: Dict, probs: np.ndarray) -> Dict:
        """Analyze battery health from features."""
        health_score = 1.0
        issues = []
        
        # Voltage check
        voltage = features.get('battery_voltage', 11.1)
        if voltage < 10.5:
            issues.append("Low voltage - RTH recommended")
            health_score *= 0.6
        elif voltage < 11.0:
            issues.append("Voltage approaching safety limit")
            health_score *= 0.8
        
        # Current check
        current = features.get('battery_current', 0)
        if current > 50:
            issues.append("High discharge rate")
            health_score *= 0.9
        
        # Remaining capacity
        remaining = features.get('battery_remaining', 100)
        if remaining < 20:
            issues.append("Low battery capacity")
            health_score *= 0.7
        
        status = 'CRITICAL' if health_score < 0.4 else 'WARNING' if health_score < 0.7 else 'HEALTHY'
        
        return {
            'health_score': float(health_score),
            'status': status,
            'voltage': voltage,
            'remaining_percent': remaining,
            'issues': issues,
            'flight_time_remaining': self._estimate_flight_time(voltage, remaining),
            'rul_estimate': self._estimate_rul(health_score, 'battery')
        }
    
    def _analyze_esc_health(self, features: Dict, probs: np.ndarray) -> Dict:
        """Analyze ESC health from features."""
        health_score = 1.0
        issues = []
        
        # Temperature check
        esc_temp = features.get('esc_temp', 50)
        if esc_temp > 90:
            issues.append("ESC overheating")
            health_score *= 0.7
        elif esc_temp > 75:
            issues.append("ESC running hot")
            health_score *= 0.85
        
        # Throttle saturation
        throttle = features.get('throttle', 0)
        if throttle > 90:
            issues.append("Throttle near maximum")
            health_score *= 0.9
        
        status = 'CRITICAL' if health_score < 0.4 else 'WARNING' if health_score < 0.7 else 'HEALTHY'
        
        return {
            'health_score': float(health_score),
            'status': status,
            'temperature': esc_temp,
            'issues': issues,
            'rul_estimate': self._estimate_rul(health_score, 'esc')
        }
    
    def _analyze_sensor_health(self, features: Dict, probs: np.ndarray) -> Dict:
        """Analyze sensor (gyro, accel, GPS) health."""
        health_score = 1.0
        issues = []
        
        # GPS health
        satellites = features.get('gps_satellites', 12)
        if satellites < 6:
            issues.append("Poor GPS fix")
            health_score *= 0.8
        
        # Gyro drift check
        gyro_values = [features.get(f'gyro_{axis}', 0) for axis in ['x', 'y', 'z']]
        if any(abs(v) > 500 for v in gyro_values):
            issues.append("Gyro reading abnormal")
            health_score *= 0.85
        
        status = 'CRITICAL' if health_score < 0.5 else 'WARNING' if health_score < 0.75 else 'HEALTHY'
        
        return {
            'health_score': float(health_score),
            'status': status,
            'gps_satellites': satellites,
            'issues': issues
        }
    
    def _analyze_propeller_health(self, features: Dict, probs: np.ndarray) -> Dict:
        """Analyze propeller and vibration health."""
        # Use vibration failure probability
        health_score = 1.0 - probs[2] * 0.9 if len(probs) > 2 else 1.0
        issues = []
        
        # Vibration magnitude
        vib_mag = features.get('vibration_magnitude', 0)
        if vib_mag > 1.5:
            issues.append("Severe vibration detected")
            health_score *= 0.6
        elif vib_mag > 1.0:
            issues.append("Elevated vibration")
            health_score *= 0.8
        
        status = 'CRITICAL' if health_score < 0.4 else 'WARNING' if health_score < 0.7 else 'HEALTHY'
        
        return {
            'health_score': float(health_score),
            'status': status,
            'vibration_level': vib_mag,
            'issues': issues,
            'rul_estimate': self._estimate_rul(health_score, 'propeller')
        }
    
    def _extract_component_features(self, features: Dict, component: str) -> Dict:
        """Extract features relevant to specific component."""
        relevant_keys = self.component_features.get(component, [])
        return {k: v for k, v in features.items() if any(rk in k for rk in relevant_keys)}
    
    def _calculate_overall_risk(self, component_health: Dict) -> float:
        """Calculate overall flight risk from component health scores."""
        health_scores = [
            comp['health_score'] for comp in component_health.values()
            if 'health_score' in comp
        ]
        
        if not health_scores:
            return 0.5
        
        # Weighted average (worst component has more weight)
        min_health = min(health_scores)
        avg_health = np.mean(health_scores)
        
        overall_risk = 1.0 - (min_health * 0.6 + avg_health * 0.4)
        return float(overall_risk)
    
    def _estimate_rul(self, health_score: float, component: str) -> str:
        """Estimate Remaining Useful Life."""
        if health_score > 0.8:
            return ">30 minutes"
        elif health_score > 0.6:
            return "10-30 minutes"
        elif health_score > 0.4:
            return "5-10 minutes"
        else:
            return "<5 minutes"
    
    def _estimate_flight_time(self, voltage: float, remaining_percent: float) -> str:
        """Estimate remaining flight time from battery status."""
        # Simple estimation (would be enhanced with actual flight model)
        minutes = (remaining_percent / 100) * 25  # Assume 25min max flight time
        voltage_factor = max(0, (voltage - 10.0) / 2.0)
        adjusted_minutes = minutes * voltage_factor
        
        if adjusted_minutes < 2:
            return "<2 minutes - LAND NOW"
        elif adjusted_minutes < 5:
            return f"{int(adjusted_minutes)} minutes - RTH recommended"
        else:
            return f"~{int(adjusted_minutes)} minutes"
    
    def _calculate_degradation_rate(self, component: str, current_health: float) -> str:
        """Calculate health degradation rate."""
        if component not in self.health_history:
            self.health_history[component] = []
        
        self.health_history[component].append(current_health)
        
        if len(self.health_history[component]) < 5:
            return "insufficient data"
        
        # Calculate trend
        recent = self.health_history[component][-5:]
        degradation = recent[0] - recent[-1]
        
        if degradation > 0.1:
            return "rapid decline"
        elif degradation > 0.05:
            return "moderate decline"
        elif degradation < -0.05:
            return "improving"
        else:
            return "stable"
    
    def _generate_comprehensive_recommendations(self,
                                                component_health: Dict,
                                                overall_risk: float,
                                                predicted_type: str) -> List[str]:
        """Generate comprehensive flight recommendations."""
        recommendations = []
        
        # Critical overall recommendations
        if overall_risk > 0.7:
            recommendations.append("🚨 HIGH RISK: Initiate emergency landing immediately")
            recommendations.append("🚨 Avoid all aggressive maneuvers")
        elif overall_risk > 0.5:
            recommendations.append("⚠️ ELEVATED RISK: Return to home recommended")
            recommendations.append("⚠️ Reduce flight envelope")
        
        # Component-specific recommendations
        for component, health in component_health.items():
            if health['status'] == 'CRITICAL':
                recommendations.extend(self._get_critical_actions(component, health))
            elif health['status'] == 'WARNING':
                recommendations.extend(self._get_warning_actions(component, health))
        
        # Specific failure type recommendations
        if predicted_type == 'motor_failure':
            recommendations.append("⚙️ Motor failure predicted - reduce throttle aggressiveness")
        elif predicted_type == 'vibration_anomaly':
            recommendations.append("🔧 Vibration anomaly - check propellers after landing")
        
        return recommendations
    
    def _get_critical_actions(self, component: str, health: Dict) -> List[str]:
        """Get critical actions for component."""
        actions = []
        if component == 'motor':
            actions.append(f"⚙️ {component.upper()}: Land immediately - {', '.join(health['issues'])}")
        elif component == 'battery':
            actions.append(f"🔋 {component.upper()}: Critical power - RTH or nearest landing zone")
        elif component == 'esc':
            actions.append(f"⚡ {component.upper()}: Thermal issue - reduce throttle, land ASAP")
        return actions
    
    def _get_warning_actions(self, component: str, health: Dict) -> List[str]:
        """Get warning actions for component."""
        actions = []
        if component == 'motor':
            actions.append(f"⚙️ {component.upper()}: Monitor closely - reduce max throttle 20%")
        elif component == 'battery':
            actions.append(f"🔋 {component.upper()}: Plan return within 3 minutes")
        elif component == 'propeller':
            actions.append(f"🚁 {component.upper()}: Reduce speed, inspect after flight")
        return actions
    
    def _get_top_features(self, features: pd.DataFrame, top_n: int = 5) -> Dict:
        """Get most important features for current prediction."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
        
        # Get top N
        indices = np.argsort(importances)[-top_n:][::-1]
        
        return {
            feature_names[i]: float(importances[i])
            for i in indices
        }
