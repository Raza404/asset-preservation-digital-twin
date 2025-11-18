"""
Hybrid Decision Engine - Combines LSTM and Random Forest for intelligent flight optimization.
Provides real-time recommendations on throttle, speed, and trajectory adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from backend.src.ml.lstm_failure_predictor import LSTMFailurePredictor
from backend.src.ml.enhanced_rf_predictor import EnhancedRandomForestPredictor


@dataclass
class FlightAction:
    """Represents a recommended flight action."""
    action_type: str  # 'throttle', 'speed', 'altitude', 'rtl', 'land'
    severity: str  # 'critical', 'warning', 'advisory'
    value: Optional[float]  # Target value (e.g., throttle %)
    reason: str  # Why this action is recommended
    component: str  # Which component triggered this
    time_critical: bool  # Must be executed immediately


class HybridDecisionEngine:
    """
    Intelligent decision engine combining LSTM and Random Forest predictions.
    
    Workflow:
    1. Random Forest: Fast anomaly detection (<50ms)
    2. LSTM: Deep analysis if RF detects issues (100-200ms)
    3. Decision Fusion: Combine both outputs
    4. Action Generation: Create specific flight commands
    """
    
    def __init__(self, 
                 rf_model_path: Optional[str] = None,
                 lstm_model_path: Optional[str] = None):
        """
        Initialize hybrid decision engine.
        
        Args:
            rf_model_path: Path to Random Forest model
            lstm_model_path: Path to LSTM model
        """
        self.rf_predictor = EnhancedRandomForestPredictor(rf_model_path)
        self.lstm_predictor = LSTMFailurePredictor(model_path=lstm_model_path)
        
        self.telemetry_buffer = []  # Store recent telemetry for LSTM
        self.buffer_size = 30  # 3 seconds at 10Hz
        
        self.action_history = []  # Track recommended actions
        self.current_flight_state = {
            'throttle': 50.0,
            'speed': 10.0,
            'altitude': 50.0,
            'battery_voltage': 11.8,
            'mode': 'normal'
        }
    
    def process_telemetry(self, telemetry: Dict) -> Dict:
        """
        Process incoming telemetry and generate recommendations.
        
        Args:
            telemetry: Current telemetry data point
        
        Returns:
            Comprehensive analysis with actions
        """
        # Add to buffer for LSTM
        self.telemetry_buffer.append(telemetry)
        if len(self.telemetry_buffer) > self.buffer_size:
            self.telemetry_buffer.pop(0)
        
        # Update flight state
        self._update_flight_state(telemetry)
        
        # Stage 1: Fast RF prediction
        rf_analysis = self._run_rf_analysis(telemetry)
        
        # Stage 2: LSTM analysis if needed
        lstm_analysis = None
        if self._should_run_lstm(rf_analysis):
            lstm_analysis = self._run_lstm_analysis()
        
        # Stage 3: Fuse predictions
        fused_analysis = self._fuse_predictions(rf_analysis, lstm_analysis)
        
        # Stage 4: Generate flight actions
        actions = self._generate_flight_actions(fused_analysis)
        
        # Store action history
        self.action_history.extend(actions)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        return {
            'rf_analysis': rf_analysis,
            'lstm_analysis': lstm_analysis,
            'fused_analysis': fused_analysis,
            'recommended_actions': [self._action_to_dict(a) for a in actions],
            'flight_state': self.current_flight_state.copy(),
            'risk_level': fused_analysis['overall_risk'],
            'requires_immediate_action': any(a.time_critical for a in actions)
        }
    
    def _run_rf_analysis(self, telemetry: Dict) -> Dict:
        """Run Random Forest analysis on current telemetry."""
        try:
            # Convert telemetry to DataFrame
            features = self._extract_features(telemetry)
            df = pd.DataFrame([features])
            
            # Get RF prediction
            analysis = self.rf_predictor.predict_with_health_analysis(df)
            return analysis
        except Exception as e:
            print(f"⚠️ RF analysis failed: {e}")
            return {'overall_risk': 0.5, 'component_health': {}, 'recommendations': []}
    
    def _run_lstm_analysis(self) -> Optional[Dict]:
        """Run LSTM analysis on telemetry sequence."""
        if len(self.telemetry_buffer) < 10:
            return None
        
        try:
            # Convert buffer to sequence
            feature_matrix = np.array([
                self._extract_features_array(t) for t in self.telemetry_buffer
            ])
            
            # Get LSTM predictions
            predictions = self.lstm_predictor.predict_component_failure(feature_matrix)
            return predictions
        except Exception as e:
            print(f"⚠️ LSTM analysis failed: {e}")
            return None
    
    def _should_run_lstm(self, rf_analysis: Dict) -> bool:
        """Determine if LSTM analysis is needed."""
        # Run LSTM if:
        # 1. RF detected elevated risk
        # 2. Any component is in warning/critical state
        # 3. Sufficient telemetry buffer
        
        if rf_analysis.get('overall_risk', 0) > 0.4:
            return True
        
        component_health = rf_analysis.get('component_health', {})
        for comp_status in component_health.values():
            if comp_status.get('status') in ['WARNING', 'CRITICAL']:
                return True
        
        return len(self.telemetry_buffer) >= 20
    
    def _fuse_predictions(self, 
                         rf_analysis: Dict, 
                         lstm_analysis: Optional[Dict]) -> Dict:
        """
        Fuse RF and LSTM predictions into comprehensive assessment.
        
        Strategy:
        - RF provides instant detection
        - LSTM provides forward-looking prediction
        - Combine using weighted confidence
        """
        fused = {
            'overall_risk': rf_analysis.get('overall_risk', 0.5),
            'component_risks': {},
            'time_horizons': {},
            'confidence': 'high' if lstm_analysis else 'medium'
        }
        
        # Component-level fusion
        rf_components = rf_analysis.get('component_health', {})
        
        for component, rf_health in rf_components.items():
            component_risk = 1.0 - rf_health['health_score']
            
            # Add LSTM prediction if available
            if lstm_analysis and component in lstm_analysis:
                lstm_risk = lstm_analysis[component]['failure_probability']
                # Weighted average: RF for current, LSTM for future
                component_risk = component_risk * 0.6 + lstm_risk * 0.4
            
            fused['component_risks'][component] = {
                'current_risk': component_risk,
                'status': rf_health['status'],
                'rul': rf_health.get('rul_estimate', 'unknown'),
                'issues': rf_health.get('issues', [])
            }
            
            # Add LSTM time horizon predictions
            if lstm_analysis and component in lstm_analysis:
                fused['time_horizons'][component] = {
                    'rul_seconds': lstm_analysis[component]['rul_seconds'],
                    'rul_human': lstm_analysis[component]['rul_human']
                }
        
        # Recalculate overall risk
        if fused['component_risks']:
            max_risk = max(c['current_risk'] for c in fused['component_risks'].values())
            avg_risk = np.mean([c['current_risk'] for c in fused['component_risks'].values()])
            fused['overall_risk'] = max_risk * 0.7 + avg_risk * 0.3
        
        return fused
    
    def _generate_flight_actions(self, fused_analysis: Dict) -> List[FlightAction]:
        """
        Generate specific, actionable flight commands.
        
        Returns list of FlightAction objects with concrete values.
        """
        actions = []
        overall_risk = fused_analysis['overall_risk']
        component_risks = fused_analysis.get('component_risks', {})
        
        # Critical risk - emergency actions
        if overall_risk > 0.7:
            actions.append(FlightAction(
                action_type='land',
                severity='critical',
                value=None,
                reason='Critical system risk detected',
                component='system',
                time_critical=True
            ))
            
            actions.append(FlightAction(
                action_type='throttle',
                severity='critical',
                value=60.0,  # Max 60% throttle
                reason='Reduce stress on failing components',
                component='system',
                time_critical=True
            ))
        
        # High risk - return to home
        elif overall_risk > 0.5:
            actions.append(FlightAction(
                action_type='rtl',
                severity='warning',
                value=None,
                reason='Elevated risk - return to home recommended',
                component='system',
                time_critical=False
            ))
        
        # Component-specific actions
        for component, risk_info in component_risks.items():
            risk = risk_info['current_risk']
            
            if component == 'motor' and risk > 0.5:
                actions.append(FlightAction(
                    action_type='throttle',
                    severity='warning' if risk < 0.7 else 'critical',
                    value=70.0 if risk < 0.7 else 60.0,
                    reason=f"Motor health degraded: {', '.join(risk_info['issues'])}",
                    component='motor',
                    time_critical=risk > 0.7
                ))
            
            if component == 'battery' and risk > 0.6:
                # Calculate safe return speed
                safe_speed = self._calculate_safe_return_speed(risk)
                actions.append(FlightAction(
                    action_type='speed',
                    severity='warning' if risk < 0.8 else 'critical',
                    value=safe_speed,
                    reason=f"Battery conservation required",
                    component='battery',
                    time_critical=risk > 0.8
                ))
            
            if component == 'propeller' and risk > 0.5:
                actions.append(FlightAction(
                    action_type='speed',
                    severity='warning',
                    value=8.0,  # Reduce to 8 m/s
                    reason=f"Vibration detected - reduce speed",
                    component='propeller',
                    time_critical=False
                ))
            
            if component == 'esc' and risk > 0.6:
                actions.append(FlightAction(
                    action_type='throttle',
                    severity='warning',
                    value=75.0,
                    reason=f"ESC thermal stress - reduce load",
                    component='esc',
                    time_critical=False
                ))
        
        # De-duplicate and prioritize
        actions = self._prioritize_actions(actions)
        
        return actions
    
    def _prioritize_actions(self, actions: List[FlightAction]) -> List[FlightAction]:
        """Prioritize and de-duplicate actions."""
        # Sort by severity and time criticality
        priority_map = {'critical': 3, 'warning': 2, 'advisory': 1}
        
        actions.sort(key=lambda a: (
            -priority_map[a.severity],
            -int(a.time_critical),
            a.action_type
        ))
        
        # Remove duplicates (keep most severe)
        seen_types = set()
        unique_actions = []
        
        for action in actions:
            if action.action_type not in seen_types:
                unique_actions.append(action)
                seen_types.add(action.action_type)
        
        return unique_actions
    
    def _calculate_safe_return_speed(self, battery_risk: float) -> float:
        """Calculate optimal return speed to conserve battery."""
        # Lower speed = less power consumption
        base_speed = 12.0  # m/s
        
        if battery_risk > 0.8:
            return 6.0  # Critical - very slow
        elif battery_risk > 0.6:
            return 8.0  # Warning - moderate
        else:
            return 10.0  # Advisory - slight reduction
    
    def _extract_features(self, telemetry: Dict) -> Dict:
        """Extract features from telemetry for RF model."""
        # This would extract the same features used during training
        # For now, return basic features
        return {
            'battery_voltage': telemetry.get('battery_voltage', 11.1),
            'battery_remaining': telemetry.get('battery_remaining', 50),
            'throttle': telemetry.get('throttle', 50),
            'vibration_magnitude': telemetry.get('vibration_magnitude', 0.5),
            'motor_temp': telemetry.get('motor_temp', 60),
            'esc_temp': telemetry.get('esc_temp', 55),
            'gps_satellites': telemetry.get('gps_satellites', 12),
            'gyro_x': telemetry.get('gyro_x', 0),
            'gyro_y': telemetry.get('gyro_y', 0),
            'gyro_z': telemetry.get('gyro_z', 0),
            'accel_x': telemetry.get('accel_x', 0),
            'accel_y': telemetry.get('accel_y', 0),
            'accel_z': telemetry.get('accel_z', 9.81),
        }
    
    def _extract_features_array(self, telemetry: Dict) -> np.ndarray:
        """Extract features as numpy array for LSTM."""
        features = self._extract_features(telemetry)
        return np.array(list(features.values()))
    
    def _update_flight_state(self, telemetry: Dict):
        """Update current flight state."""
        self.current_flight_state.update({
            'throttle': telemetry.get('throttle', self.current_flight_state['throttle']),
            'speed': telemetry.get('ground_speed', self.current_flight_state['speed']),
            'altitude': telemetry.get('altitude', self.current_flight_state['altitude']),
            'battery_voltage': telemetry.get('battery_voltage', self.current_flight_state['battery_voltage'])
        })
    
    def _action_to_dict(self, action: FlightAction) -> Dict:
        """Convert FlightAction to dictionary."""
        return {
            'action_type': action.action_type,
            'severity': action.severity,
            'value': action.value,
            'reason': action.reason,
            'component': action.component,
            'time_critical': action.time_critical,
            'display': self._format_action(action)
        }
    
    def _format_action(self, action: FlightAction) -> str:
        """Format action for display."""
        emoji_map = {
            'critical': '🚨',
            'warning': '⚠️',
            'advisory': 'ℹ️'
        }
        
        emoji = emoji_map.get(action.severity, '•')
        
        if action.action_type == 'throttle':
            return f"{emoji} THROTTLE: Limit to {action.value}% - {action.reason}"
        elif action.action_type == 'speed':
            return f"{emoji} SPEED: Reduce to {action.value} m/s - {action.reason}"
        elif action.action_type == 'land':
            return f"{emoji} LAND IMMEDIATELY - {action.reason}"
        elif action.action_type == 'rtl':
            return f"{emoji} RETURN TO HOME - {action.reason}"
        elif action.action_type == 'altitude':
            return f"{emoji} ALTITUDE: Adjust to {action.value}m - {action.reason}"
        else:
            return f"{emoji} {action.action_type.upper()}: {action.reason}"
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary report."""
        if not self.action_history:
            return "No actions recommended - all systems nominal"
        
        recent_actions = self.action_history[-5:]
        
        report = "=== FLIGHT OPTIMIZATION SUMMARY ===\n\n"
        report += f"Current Flight State:\n"
        report += f"  Throttle: {self.current_flight_state['throttle']:.1f}%\n"
        report += f"  Speed: {self.current_flight_state['speed']:.1f} m/s\n"
        report += f"  Altitude: {self.current_flight_state['altitude']:.1f}m\n"
        report += f"  Battery: {self.current_flight_state['battery_voltage']:.2f}V\n\n"
        
        report += f"Recent Actions ({len(recent_actions)}):\n"
        for i, action in enumerate(recent_actions, 1):
            report += f"  {i}. {self._format_action(action)}\n"
        
        return report
