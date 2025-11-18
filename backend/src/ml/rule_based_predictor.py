"""
Production-Ready Rule-Based Failure Predictor
Enhanced heuristics for component health monitoring without ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ComponentHealth:
    """Component health analysis result."""
    component: str
    risk_level: float  # 0-1
    status: str  # HEALTHY, WARNING, CRITICAL
    predicted_rul: int  # seconds
    triggers: List[str]
    recommendations: List[str]
    diagnostic_info: Dict


class RuleBasedPredictor:
    """
    Production-ready rule-based component health predictor.
    Uses sophisticated heuristics for failure prediction without ML models.
    """
    
    def __init__(self):
        self.component_thresholds = {
            'battery': {'critical': 10.5, 'warning': 11.1},
            'motor_temp': {'critical': 95, 'warning': 75},
            'esc_temp': {'critical': 85, 'warning': 65},
            'vibration': {'critical': 2.0, 'warning': 1.0},
            'gps_sats': {'critical': 6, 'warning': 9}
        }
    
    def analyze_telemetry(self, telemetry_buffer: List[Dict]) -> Dict[str, ComponentHealth]:
        """
        Analyze telemetry buffer and predict component failures.
        
        Args:
            telemetry_buffer: List of recent telemetry readings (30-60 samples)
        
        Returns:
            Dictionary of component health analyses
        """
        if not telemetry_buffer:
            return {}
        
        # Convert to arrays for analysis
        df = pd.DataFrame(telemetry_buffer)
        
        results = {}
        results['battery'] = self._analyze_battery(df)
        results['motor'] = self._analyze_motor(df)
        results['esc'] = self._analyze_esc(df)
        results['sensors'] = self._analyze_sensors(df)
        results['propeller'] = self._analyze_propeller(df)
        
        return results
    
    def _analyze_battery(self, df: pd.DataFrame) -> ComponentHealth:
        """
        Enhanced battery analysis with voltage curve and discharge rate monitoring.
        """
        risk = 0.0
        triggers = []
        
        # Extract battery metrics
        voltage = df.get('battery_voltage', pd.Series([12.0]))
        remaining = df.get('battery_remaining', pd.Series([100]))
        throttle = df.get('throttle', pd.Series([50]))
        
        avg_voltage = voltage.mean()
        min_voltage = voltage.min()
        voltage_std = voltage.std()
        
        # Calculate discharge rate (V/sec)
        if len(voltage) > 1:
            time_span = len(voltage)  # Assuming 1 sample per second
            discharge_rate = (voltage.iloc[0] - voltage.iloc[-1]) / time_span
        else:
            discharge_rate = 0.0
        
        # Voltage sag detection (sudden drops)
        voltage_diffs = voltage.diff()
        max_sag = voltage_diffs.min() if len(voltage_diffs) > 0 else 0.0
        
        # Critical voltage
        if min_voltage < 10.5:
            risk = max(risk, 0.95)
            triggers.append('critical_voltage_10.5V_or_below')
        elif avg_voltage < 11.1:
            risk = max(risk, 0.70)
            triggers.append('low_voltage_warning_below_11.1V')
        
        # Rapid discharge
        if discharge_rate > 0.15:  # >0.15V per second
            risk = max(risk, 0.75)
            triggers.append('rapid_discharge_rate')
        elif discharge_rate > 0.08:
            risk = max(risk, 0.55)
            triggers.append('elevated_discharge_rate')
        
        # Voltage sag (internal resistance)
        if max_sag < -0.5:
            risk = max(risk, 0.65)
            triggers.append('voltage_sag_high_internal_resistance')
        
        # Voltage instability
        if voltage_std > 0.3:
            risk = max(risk, 0.60)
            triggers.append('unstable_voltage_pattern')
        
        # Under load analysis
        high_throttle_moments = throttle[throttle > 70]
        if len(high_throttle_moments) > 0:
            voltage_under_load = voltage[throttle > 70].mean()
            if voltage_under_load < 10.8:
                risk = max(risk, 0.80)
                triggers.append('poor_performance_under_load')
        
        # Calculate RUL
        if avg_voltage > 10.5:
            rul = int((avg_voltage - 10.5) / max(discharge_rate, 0.01) * 60)
            rul = max(0, min(rul, 3600))
        else:
            rul = 0
        
        if not triggers:
            risk = 0.15
            triggers.append('healthy')
        
        status = 'CRITICAL' if risk > 0.7 else 'WARNING' if risk > 0.4 else 'HEALTHY'
        
        return ComponentHealth(
            component='battery',
            risk_level=risk,
            status=status,
            predicted_rul=rul,
            triggers=triggers,
            recommendations=self._get_battery_recommendations(risk, triggers),
            diagnostic_info={
                'avg_voltage': float(avg_voltage),
                'min_voltage': float(min_voltage),
                'discharge_rate_V_per_sec': float(discharge_rate),
                'voltage_stability': float(voltage_std),
                'max_voltage_sag': float(max_sag)
            }
        )
    
    def _analyze_motor(self, df: pd.DataFrame) -> ComponentHealth:
        """
        Enhanced motor analysis with temperature trends and vibration correlation.
        """
        risk = 0.0
        triggers = []
        
        # Extract motor metrics
        temp = df.get('motor_temp', pd.Series([50.0]))
        vibration = df.get('vibration_magnitude', pd.Series([0.3]))
        throttle = df.get('throttle', pd.Series([50.0]))
        
        avg_temp = temp.mean()
        max_temp = temp.max()
        temp_trend = (temp.iloc[-1] - temp.iloc[0]) / len(temp) if len(temp) > 1 else 0.0
        
        avg_vibration = vibration.mean()
        max_vibration = vibration.max()
        vibration_std = vibration.std()
        
        avg_throttle = throttle.mean()
        
        # Temperature analysis
        if max_temp > 95:
            risk = max(risk, 0.95)
            triggers.append('critical_motor_temperature_95C_plus')
        elif avg_temp > 85:
            risk = max(risk, 0.75)
            triggers.append('very_high_temperature_85C_plus')
        elif avg_temp > 70:
            risk = max(risk, 0.50)
            triggers.append('elevated_temperature_70C_plus')
        
        # Rapid heating
        if temp_trend > 0.5:  # >0.5°C per second
            risk = max(risk, 0.65)
            triggers.append('rapid_temperature_rise')
        
        # Vibration analysis
        if max_vibration > 2.0:
            risk = max(risk, 0.85)
            triggers.append('severe_vibration_above_2.0')
        elif avg_vibration > 1.2:
            risk = max(risk, 0.70)
            triggers.append('high_vibration_above_1.2')
        elif avg_vibration > 0.7:
            risk = max(risk, 0.45)
            triggers.append('elevated_vibration')
        
        # Vibration instability (bearing wear)
        if vibration_std > 0.4:
            risk = max(risk, 0.60)
            triggers.append('unstable_vibration_bearing_wear')
        
        # Efficiency check (hot at low throttle = problem)
        if avg_temp > 75 and avg_throttle < 40:
            risk = max(risk, 0.55)
            triggers.append('inefficient_operation_hot_at_low_throttle')
        
        # Combined thermal-mechanical stress
        if avg_temp > 70 and avg_vibration > 0.8:
            risk = max(risk, 0.75)
            triggers.append('combined_thermal_mechanical_stress')
        
        # RUL calculation
        if risk > 0.7:
            rul = 120  # 2 minutes
        elif risk > 0.4:
            rul = 300  # 5 minutes
        else:
            rul = 900  # 15 minutes
        
        if not triggers:
            risk = 0.10
            triggers.append('healthy')
        
        status = 'CRITICAL' if risk > 0.7 else 'WARNING' if risk > 0.4 else 'HEALTHY'
        
        return ComponentHealth(
            component='motor',
            risk_level=risk,
            status=status,
            predicted_rul=rul,
            triggers=triggers,
            recommendations=self._get_motor_recommendations(risk, triggers),
            diagnostic_info={
                'avg_temp_C': float(avg_temp),
                'max_temp_C': float(max_temp),
                'temp_trend_C_per_sec': float(temp_trend),
                'avg_vibration': float(avg_vibration),
                'vibration_stability': float(vibration_std)
            }
        )
    
    def _analyze_esc(self, df: pd.DataFrame) -> ComponentHealth:
        """
        ESC (Electronic Speed Controller) health analysis.
        """
        risk = 0.0
        triggers = []
        
        # Extract ESC metrics
        esc_temp = df.get('esc_temp', pd.Series([45.0]))
        motor_temp = df.get('motor_temp', pd.Series([50.0]))
        throttle = df.get('throttle', pd.Series([50.0]))
        
        avg_esc_temp = esc_temp.mean()
        max_esc_temp = esc_temp.max()
        esc_temp_trend = (esc_temp.iloc[-1] - esc_temp.iloc[0]) / len(esc_temp) if len(esc_temp) > 1 else 0.0
        
        avg_throttle = throttle.mean()
        throttle_changes = throttle.diff().abs().sum() / len(throttle) if len(throttle) > 1 else 0.0
        
        # ESC temperature analysis
        if max_esc_temp > 85:
            risk = max(risk, 0.90)
            triggers.append('critical_esc_temperature_85C_plus')
        elif avg_esc_temp > 75:
            risk = max(risk, 0.65)
            triggers.append('high_esc_temperature_75C_plus')
        elif avg_esc_temp > 60:
            risk = max(risk, 0.40)
            triggers.append('elevated_esc_temperature')
        
        # ESC heating rate
        if esc_temp_trend > 0.8:
            risk = max(risk, 0.70)
            triggers.append('rapid_esc_heating')
        
        # ESC-Motor temperature differential
        temp_diff = abs(esc_temp.mean() - motor_temp.mean())
        if temp_diff > 20:
            risk = max(risk, 0.55)
            triggers.append('unusual_esc_motor_temp_differential')
        
        # High throttle stress
        if avg_throttle > 85 and throttle_changes > 5:
            risk = max(risk, 0.50)
            triggers.append('high_stress_operation')
        
        # RUL calculation
        if risk > 0.7:
            rul = 90
        elif risk > 0.4:
            rul = 240
        else:
            rul = 600
        
        if not triggers:
            risk = 0.10
            triggers.append('healthy')
        
        status = 'CRITICAL' if risk > 0.7 else 'WARNING' if risk > 0.4 else 'HEALTHY'
        
        return ComponentHealth(
            component='esc',
            risk_level=risk,
            status=status,
            predicted_rul=rul,
            triggers=triggers,
            recommendations=self._get_esc_recommendations(risk, triggers),
            diagnostic_info={
                'avg_esc_temp_C': float(avg_esc_temp),
                'max_esc_temp_C': float(max_esc_temp),
                'temp_trend': float(esc_temp_trend),
                'esc_motor_diff': float(temp_diff)
            }
        )
    
    def _analyze_sensors(self, df: pd.DataFrame) -> ComponentHealth:
        """
        Sensor health analysis (GPS, IMU quality).
        """
        risk = 0.0
        triggers = []
        
        # Extract sensor metrics
        gps_sats = df.get('gps_satellites', pd.Series([12.0]))
        gyro_x = df.get('gyro_x', pd.Series([0.0]))
        gyro_y = df.get('gyro_y', pd.Series([0.0]))
        gyro_z = df.get('gyro_z', pd.Series([0.0]))
        
        avg_gps_sats = gps_sats.mean()
        min_gps_sats = gps_sats.min()
        gps_stability = gps_sats.std()
        
        # IMU noise analysis
        gyro_noise = np.sqrt(gyro_x.std()**2 + gyro_y.std()**2 + gyro_z.std()**2)
        
        # GPS quality
        if min_gps_sats < 6:
            risk = max(risk, 0.85)
            triggers.append('insufficient_gps_satellites_below_6')
        elif avg_gps_sats < 9:
            risk = max(risk, 0.55)
            triggers.append('poor_gps_quality_below_9')
        
        # GPS instability
        if gps_stability > 2.0:
            risk = max(risk, 0.60)
            triggers.append('unstable_gps_signal')
        
        # IMU noise
        if gyro_noise > 50:
            risk = max(risk, 0.70)
            triggers.append('excessive_gyro_noise')
        elif gyro_noise > 30:
            risk = max(risk, 0.45)
            triggers.append('elevated_gyro_noise')
        
        # RUL calculation
        if risk > 0.7:
            rul = 60
        elif risk > 0.4:
            rul = 300
        else:
            rul = 1200
        
        if not triggers:
            risk = 0.10
            triggers.append('healthy')
        
        status = 'CRITICAL' if risk > 0.7 else 'WARNING' if risk > 0.4 else 'HEALTHY'
        
        return ComponentHealth(
            component='sensors',
            risk_level=risk,
            status=status,
            predicted_rul=rul,
            triggers=triggers,
            recommendations=self._get_sensor_recommendations(risk, triggers),
            diagnostic_info={
                'avg_gps_satellites': float(avg_gps_sats),
                'gps_stability': float(gps_stability),
                'gyro_noise_level': float(gyro_noise)
            }
        )
    
    def _analyze_propeller(self, df: pd.DataFrame) -> ComponentHealth:
        """
        Propeller health analysis based on vibration patterns.
        """
        risk = 0.0
        triggers = []
        
        # Extract metrics
        vibration = df.get('vibration_magnitude', pd.Series([0.3]))
        throttle = df.get('throttle', pd.Series([50.0]))
        altitude = df.get('altitude', pd.Series([50.0]))
        
        avg_vibration = vibration.mean()
        vibration_pattern = vibration.std()
        avg_throttle = throttle.mean()
        
        # Climb rate calculation
        altitude_rate = altitude.diff().mean() if len(altitude) > 1 else 0.0
        
        # High frequency vibration
        if avg_vibration > 1.5:
            risk = max(risk, 0.80)
            triggers.append('severe_vibration_unbalanced_prop')
        elif avg_vibration > 1.0:
            risk = max(risk, 0.60)
            triggers.append('high_vibration_possible_damage')
        
        # Vibration pattern irregularity
        if vibration_pattern > 0.5:
            risk = max(risk, 0.65)
            triggers.append('irregular_vibration_pattern')
        
        # Efficiency check (high throttle, low climb = damaged prop)
        if avg_throttle > 70 and abs(altitude_rate) < 0.5:
            risk = max(risk, 0.70)
            triggers.append('low_efficiency_possible_prop_damage')
        
        # RUL calculation
        if risk > 0.7:
            rul = 120
        elif risk > 0.4:
            rul = 360
        else:
            rul = 900
        
        if not triggers:
            risk = 0.10
            triggers.append('healthy')
        
        status = 'CRITICAL' if risk > 0.7 else 'WARNING' if risk > 0.4 else 'HEALTHY'
        
        return ComponentHealth(
            component='propeller',
            risk_level=risk,
            status=status,
            predicted_rul=rul,
            triggers=triggers,
            recommendations=self._get_propeller_recommendations(risk, triggers),
            diagnostic_info={
                'avg_vibration': float(avg_vibration),
                'vibration_pattern': float(vibration_pattern),
                'efficiency_ratio': float(altitude_rate / max(avg_throttle, 1))
            }
        )
    
    # Recommendation generators
    
    def _get_battery_recommendations(self, risk: float, triggers: List[str]) -> List[str]:
        if risk > 0.7:
            recs = ["🔋 IMMEDIATE LANDING REQUIRED"]
            if 'critical_voltage' in str(triggers):
                recs.append("🔋 Battery voltage critically low - imminent shutdown risk")
            if 'rapid_discharge' in str(triggers):
                recs.append("🔋 Rapid power drain detected")
            recs.extend([
                "🔋 Reduce throttle to minimum safe level",
                "🔋 Activate return-to-home immediately",
                "🔋 Disable all non-essential systems"
            ])
            return recs
        elif risk > 0.4:
            return [
                "🔋 Land within 2-3 minutes",
                "🔋 Reduce power consumption - lower speed",
                "🔋 Avoid climbing - maintain or descend altitude",
                "🔋 Monitor voltage every 10 seconds",
                "🔋 Plan most direct return route"
            ]
        else:
            return ["🔋 Battery healthy", "🔋 Continue normal operation"]
    
    def _get_motor_recommendations(self, risk: float, triggers: List[str]) -> List[str]:
        if risk > 0.7:
            recs = ["⚠️ CRITICAL: Motor failure imminent"]
            if 'critical_motor_temperature' in str(triggers):
                recs.append("⚠️ Motor overheating - thermal damage risk")
            if 'severe_vibration' in str(triggers):
                recs.append("⚠️ Severe vibration - bearing or shaft failure likely")
            recs.extend([
                "⚠️ Land immediately",
                "⚠️ Reduce throttle to 40% maximum",
                "⚠️ Avoid all aggressive maneuvers",
                "⚠️ Post-flight: Inspect motor bearings and shaft"
            ])
            return recs
        elif risk > 0.4:
            return [
                "⚡ Motor stress detected - reduce workload",
                "⚡ Limit throttle to 70%",
                "⚡ Avoid rapid direction changes",
                "⚡ Monitor temperature closely",
                "⚡ Land within 5 minutes"
            ]
        else:
            return ["⚡ Motor healthy", "⚡ Continue normal operation"]
    
    def _get_esc_recommendations(self, risk: float, triggers: List[str]) -> List[str]:
        if risk > 0.7:
            return [
                "⚡ CRITICAL: ESC overheating",
                "⚡ Reduce throttle immediately to 50%",
                "⚡ Use smooth throttle inputs only",
                "⚡ Avoid sudden throttle changes",
                "⚡ Land as soon as safe",
                "⚡ Post-flight: Check ESC cooling and solder joints"
            ]
        elif risk > 0.4:
            return [
                "⚡ ESC running hot - reduce load",
                "⚡ Limit throttle to 70%",
                "⚡ Avoid prolonged high-throttle",
                "⚡ Use gentler acceleration",
                "⚡ Consider landing to cool down"
            ]
        else:
            return ["⚡ ESC healthy", "⚡ Continue normal operation"]
    
    def _get_sensor_recommendations(self, risk: float, triggers: List[str]) -> List[str]:
        if risk > 0.7:
            recs = ["📡 CRITICAL: Sensor degradation"]
            if 'insufficient_gps' in str(triggers):
                recs.append("📡 GPS unreliable - manual flight only")
            recs.extend([
                "📡 Switch to manual/stabilized mode immediately",
                "📡 Do NOT trust GPS navigation",
                "📡 Maintain visual line of sight",
                "📡 Land immediately if possible"
            ])
            return recs
        elif risk > 0.4:
            return [
                "📡 Sensor quality degraded",
                "📡 Avoid autonomous modes",
                "📡 Cross-check all readings",
                "📡 Fly in clear conditions only",
                "📡 Consider landing if degradation continues"
            ]
        else:
            return ["📡 Sensors healthy", "📡 Continue normal operation"]
    
    def _get_propeller_recommendations(self, risk: float, triggers: List[str]) -> List[str]:
        if risk > 0.7:
            return [
                "🚁 CRITICAL: Propeller damage detected",
                "🚁 Land immediately - possible prop strike",
                "🚁 Severe vibration - structural failure risk",
                "🚁 Reduce speed to minimum controllable",
                "🚁 Post-flight: Inspect all propellers for cracks/chips",
                "🚁 Replace damaged propellers before next flight"
            ]
        elif risk > 0.4:
            return [
                "🚁 Propeller condition degraded",
                "🚁 Reduce max speed by 30%",
                "🚁 Avoid aggressive maneuvers",
                "🚁 Land within 5 minutes",
                "🚁 Post-flight: Check propeller balance"
            ]
        else:
            return ["🚁 Propellers healthy", "🚁 Continue normal operation"]
