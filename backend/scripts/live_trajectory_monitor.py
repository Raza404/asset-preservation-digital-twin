"""
Live Trajectory Optimization Monitor
Integrates ML prediction, trajectory optimization, and real-time decision making
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
from datetime import datetime
from typing import Dict, Optional
import joblib
import numpy as np

from src.processing.trajectory_optimizer import (
    TrajectoryOptimizer,
    DroneState,
    Waypoint,
    OptimizationStrategy
)
from src.processing.decision_maker import RealTimeDecisionMaker, AdaptiveDecisionMaker
from src.ml.feature_engineering import UniversalFeatureExtractor, FeatureConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveOptimizationMonitor:
    """
    Real-time monitoring system that:
    1. Receives telemetry data
    2. Predicts failures using ML
    3. Optimizes trajectory
    4. Makes autonomous decisions
    """
    
    def __init__(
        self,
        ml_model_path: str = None,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        auto_execute: bool = False
    ):
        # Load ML model
        if ml_model_path and os.path.exists(ml_model_path):
            logger.info(f"Loading ML model from {ml_model_path}")
            self.ml_model = joblib.load(ml_model_path)
        else:
            logger.warning("No ML model loaded - using mock predictions")
            self.ml_model = None
        
        # Initialize feature extractor
        self.feature_extractor = UniversalFeatureExtractor(
            FeatureConfig(
                window_size_seconds=1.0,
                rolling_window_size=10,
                include_statistical=True,
                include_derivative=True
            )
        )
        
        # Initialize trajectory optimizer
        self.optimizer = TrajectoryOptimizer(
            strategy=strategy,
            failure_threshold=0.25,
            battery_warning_threshold=0.30,
            battery_critical_threshold=0.20
        )
        
        # Initialize decision maker
        self.decision_maker = AdaptiveDecisionMaker(
            optimizer=self.optimizer,
            auto_execute=auto_execute,
            log_decisions=True
        )
        
        # Telemetry buffer for feature extraction
        self.telemetry_buffer = []
        self.buffer_size = 50
        
        # Statistics
        self.stats = {
            'telemetry_received': 0,
            'decisions_made': 0,
            'commands_sent': 0,
            'critical_events': 0
        }
        
    def process_telemetry_stream(self, telemetry: Dict):
        """
        Process a single telemetry message
        This would be called for each MQTT message received
        """
        self.stats['telemetry_received'] += 1
        
        # Add to buffer
        self.telemetry_buffer.append(telemetry)
        if len(self.telemetry_buffer) > self.buffer_size:
            self.telemetry_buffer.pop(0)
        
        # Predict failure probability
        failure_prob = self._predict_failure(telemetry)
        
        # Estimate component health
        component_health = self._estimate_component_health(telemetry)
        
        # Get target waypoint (would come from mission planner)
        target = self._get_target_waypoint()
        
        # Make decision
        actions, command = self.decision_maker.process_telemetry(
            telemetry,
            failure_prob,
            component_health,
            target
        )
        
        if actions:
            self.stats['decisions_made'] += 1
            self._display_decision(telemetry, failure_prob, actions, command)
        
        if command:
            self.stats['commands_sent'] += 1
            self._send_command_to_drone(command)
        
        # Check for critical events
        if failure_prob > 0.75 or telemetry.get('battery_remaining', 100) < 15:
            self.stats['critical_events'] += 1
            logger.critical(f"⚠️  CRITICAL EVENT: Failure prob={failure_prob:.1%}, Battery={telemetry.get('battery_remaining', 0):.1f}%")
    
    def _predict_failure(self, telemetry: Dict) -> float:
        """Predict failure probability using ML model"""
        if self.ml_model is None or len(self.telemetry_buffer) < 10:
            # Mock prediction based on simple heuristics
            return self._mock_failure_prediction(telemetry)
        
        try:
            # Extract features from telemetry buffer
            features = self.feature_extractor.extract_features(self.telemetry_buffer)
            
            # Predict using ML model
            if len(features) > 0:
                prediction = self.ml_model.predict_proba(features.iloc[-1:])
                return float(prediction[0][1])  # Probability of failure class
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
        
        return self._mock_failure_prediction(telemetry)
    
    def _mock_failure_prediction(self, telemetry: Dict) -> float:
        """Simple heuristic-based failure prediction"""
        risk = 0.0
        
        # Battery risk
        battery = telemetry.get('battery_remaining', 100.0)
        if battery < 20:
            risk += 0.4
        elif battery < 40:
            risk += 0.2
        
        # Voltage sag risk
        voltage = telemetry.get('battery_voltage', 12.6)
        if voltage < 13.2:  # 3.3V per cell for 4S
            risk += 0.3
        
        # Vibration risk (if available)
        accel_x = telemetry.get('accel_x', 0)
        accel_y = telemetry.get('accel_y', 0)
        accel_z = telemetry.get('accel_z', 0)
        vibration = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        if vibration > 20:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _estimate_component_health(self, telemetry: Dict) -> Dict[str, float]:
        """Estimate health of drone components"""
        health = {
            'motor': 1.0,
            'esc': 1.0,
            'battery': 1.0,
            'gps': 1.0,
            'sensors': 1.0
        }
        
        # Battery health based on voltage and remaining
        voltage = telemetry.get('battery_voltage', 12.6)
        remaining = telemetry.get('battery_remaining', 100.0) / 100.0
        
        if voltage < 13.2:
            health['battery'] = 0.6
        if voltage < 12.8:
            health['battery'] = 0.3
        
        # Motor/ESC health based on vibration
        if len(self.telemetry_buffer) > 10:
            recent_vibration = np.mean([
                np.sqrt(
                    t.get('accel_x', 0)**2 +
                    t.get('accel_y', 0)**2 +
                    t.get('accel_z', 0)**2
                )
                for t in self.telemetry_buffer[-10:]
            ])
            
            if recent_vibration > 25:
                health['motor'] = 0.5
                health['esc'] = 0.6
            elif recent_vibration > 15:
                health['motor'] = 0.7
                health['esc'] = 0.8
        
        # GPS health
        if telemetry.get('gps_satellites', 10) < 6:
            health['gps'] = 0.5
        
        return health
    
    def _get_target_waypoint(self) -> Optional[Waypoint]:
        """Get next target waypoint from mission planner"""
        # In real implementation, this would come from mission planner
        # For now, return None (hover/maintain position)
        return None
    
    def _display_decision(
        self,
        telemetry: Dict,
        failure_prob: float,
        actions: list,
        command: Optional[str]
    ):
        """Display decision information"""
        print("\n" + "="*80)
        print(f"📊 DECISION POINT - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Current state
        print(f"\n📍 Current State:")
        print(f"   Position: ({telemetry.get('latitude', 0):.6f}, {telemetry.get('longitude', 0):.6f})")
        print(f"   Altitude: {telemetry.get('altitude', 0):.1f}m")
        print(f"   Speed: {telemetry.get('ground_speed', 0):.1f}m/s")
        print(f"   Battery: {telemetry.get('battery_remaining', 0):.1f}% ({telemetry.get('battery_voltage', 0):.2f}V)")
        print(f"   Failure Probability: {failure_prob:.1%}")
        
        # Recommended actions
        print(f"\n🎯 Recommended Actions:")
        for i, action in enumerate(actions[:3], 1):
            urgency_icon = "🔴" if action.urgency > 0.8 else "🟡" if action.urgency > 0.5 else "🟢"
            print(f"   {i}. {urgency_icon} {action.action.value} (urgency: {action.urgency:.1%})")
            print(f"      Reason: {action.reason}")
            if action.estimated_benefit:
                print(f"      Benefit: {action.estimated_benefit}")
        
        # Command to execute
        if command:
            print(f"\n⚡ COMMAND TO EXECUTE: {command}")
        else:
            print(f"\n✓ No action required - continuing normal operation")
        
        print("="*80)
    
    def _send_command_to_drone(self, command: str):
        """
        Send command to drone
        In real implementation, this would publish to MQTT or send via MAVLink
        """
        logger.info(f"📤 SENDING COMMAND: {command}")
        # TODO: Implement actual command sending via MQTT/MAVLink
        # mqtt_client.publish('drone/commands', command)
    
    def run_simulation(self, duration_seconds: int = 60, interval: float = 1.0):
        """
        Run a simulation with synthetic telemetry
        Useful for testing without a real drone
        """
        logger.info(f"Starting simulation for {duration_seconds} seconds...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_seconds:
            iteration += 1
            
            # Generate synthetic telemetry
            telemetry = self._generate_synthetic_telemetry(iteration)
            
            # Process it
            self.process_telemetry_stream(telemetry)
            
            # Wait
            time.sleep(interval)
        
        # Print statistics
        self._print_statistics()
    
    def _generate_synthetic_telemetry(self, iteration: int) -> Dict:
        """Generate realistic synthetic telemetry for testing"""
        # Simulate battery drain
        battery_drain_rate = 0.5  # % per iteration
        battery = max(5, 100 - (iteration * battery_drain_rate))
        
        # Simulate voltage sag
        voltage = 16.8 * (battery / 100) - 0.5 if battery < 50 else 16.8 * (battery / 100)
        
        # Simulate increasing vibration as battery drains (motors work harder)
        base_vibration = 8.0
        stress_vibration = (100 - battery) / 5
        vibration = base_vibration + stress_vibration + np.random.normal(0, 2)
        
        return {
            'timestamp': time.time(),
            'latitude': 40.7128 + np.random.normal(0, 0.0001),
            'longitude': -74.0060 + np.random.normal(0, 0.0001),
            'altitude': 50.0 + np.random.normal(0, 2),
            'ground_speed': 8.0 + np.random.normal(0, 1),
            'heading': 90.0,
            'battery_voltage': voltage + np.random.normal(0, 0.1),
            'battery_remaining': battery,
            'accel_x': np.random.normal(0, vibration),
            'accel_y': np.random.normal(0, vibration),
            'accel_z': 9.81 + np.random.normal(0, vibration),
            'gyro_x': np.random.normal(0, 0.5),
            'gyro_y': np.random.normal(0, 0.5),
            'gyro_z': np.random.normal(0, 0.5),
            'gps_satellites': 12,
            'wind_speed': 3.5
        }
    
    def _print_statistics(self):
        """Print session statistics"""
        print("\n" + "="*80)
        print("📈 SESSION STATISTICS")
        print("="*80)
        for key, value in self.stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        summary = self.decision_maker.get_decision_summary()
        print(f"\n   Total Decisions Made: {summary['total_decisions']}")
        print(f"\n   Action Distribution:")
        for action, count in summary.get('action_distribution', {}).items():
            print(f"      {action}: {count}")
        print("="*80)


def main():
    """Run the live optimization monitor"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  LIVE TRAJECTORY OPTIMIZATION MONITOR                        ║
║                                                                              ║
║  Real-time failure prediction and autonomous trajectory optimization        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize monitor
    ml_model_path = os.path.join('..', 'src', 'ml', 'failure_model.joblib')
    
    monitor = LiveOptimizationMonitor(
        ml_model_path=ml_model_path,
        strategy=OptimizationStrategy.BALANCED,
        auto_execute=False  # Set to True to send commands automatically
    )
    
    # Run simulation
    print("\n🚁 Starting simulation with synthetic telemetry...")
    print("   Watch how the system responds to degrading conditions!\n")
    
    monitor.run_simulation(duration_seconds=120, interval=2.0)
    
    # Export decision log
    log_path = os.path.join('..', '..', 'data', 'logs', f'decisions_{int(time.time())}.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    monitor.decision_maker.export_decision_log(log_path)
    
    print(f"\n✅ Decision log saved to: {log_path}")


if __name__ == "__main__":
    main()
