"""
Example: Custom Usage of the Digital Twin System
Demonstrates how to customize and extend the system
"""

import numpy as np
from digital_twin_engine import DigitalTwinEngine
from telemetry_simulator import TelemetrySimulator
from anomaly_detection import AnomalyDetector
from path_planning import AdaptivePathPlanner


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize
    simulator = TelemetrySimulator()
    engine = DigitalTwinEngine()
    
    # Train with historical data
    training_data = simulator.generate_training_dataset(500)
    engine.initialize(training_data)
    
    # Start mission
    engine.start_mission((0, 0, 50), (100, 100, 50))
    
    # Process single telemetry update
    telemetry = simulator.generate_normal_telemetry(1)[0]
    update = engine.update_telemetry(telemetry, (10, 10, 50))
    
    print(f"\nRisk Level: {update['risk_assessment']['risk_level']}")
    print(f"Anomaly Score: {update['risk_assessment']['anomaly_score']:.3f}")
    
    engine.stop_mission()
    print()


def example_custom_anomaly_detector():
    """Example of using anomaly detector standalone"""
    print("=" * 60)
    print("Example 2: Standalone Anomaly Detection")
    print("=" * 60)
    
    simulator = TelemetrySimulator()
    detector = AnomalyDetector(contamination=0.15)
    
    # Train detector
    training_data = simulator.generate_training_dataset(500)
    detector.train(training_data)
    
    # Test on normal data
    normal_sample = simulator.generate_normal_telemetry(1)[0]
    risk = detector.detect_failure_risk(normal_sample)
    print(f"\nNormal Sample:")
    print(f"  Risk: {risk['risk_level']}")
    print(f"  Score: {risk['anomaly_score']:.3f}")
    print(f"  Recommendation: {risk['recommendation']}")
    
    # Test on anomalous data
    anomalous_sample = simulator.generate_anomalous_telemetry(1, 'battery')[0]
    risk = detector.detect_failure_risk(anomalous_sample)
    print(f"\nAnomalous Sample (Low Battery):")
    print(f"  Risk: {risk['risk_level']}")
    print(f"  Score: {risk['anomaly_score']:.3f}")
    print(f"  Recommendation: {risk['recommendation']}")
    print()


def example_path_planning():
    """Example of path planning with different risk levels"""
    print("=" * 60)
    print("Example 3: Adaptive Path Planning")
    print("=" * 60)
    
    planner = AdaptivePathPlanner()
    
    current_pos = (50, 50, 50)
    destination = (100, 100, 50)
    
    # Test different risk levels
    risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    for risk_level in risk_levels:
        risk_assessment = {'risk_level': risk_level}
        path = planner.optimize_trajectory(current_pos, risk_assessment, destination)
        metrics = planner.calculate_path_metrics(path)
        
        print(f"\n{risk_level} Risk Path:")
        print(f"  Distance: {metrics['total_distance']:.2f}m")
        print(f"  Waypoints: {metrics['num_waypoints']}")
        print(f"  Final altitude: {path[-1][2]:.2f}m")
    print()


def example_real_time_monitoring():
    """Example of continuous real-time monitoring"""
    print("=" * 60)
    print("Example 4: Real-Time Monitoring")
    print("=" * 60)
    
    simulator = TelemetrySimulator()
    engine = DigitalTwinEngine()
    
    # Initialize
    training_data = simulator.generate_training_dataset(500)
    engine.initialize(training_data)
    
    # Start mission
    engine.start_mission((0, 0, 50), (50, 50, 50))
    
    # Simulate 10 time steps
    print("\nMonitoring flight:")
    for i in range(10):
        # Generate telemetry (gradually degrading)
        if i < 5:
            telemetry = simulator.generate_normal_telemetry(1)[0]
        else:
            # Introduce anomaly
            telemetry = simulator.generate_anomalous_telemetry(1, 'temperature')[0]
        
        position = (i * 5, i * 5, 50)
        update = engine.update_telemetry(telemetry, position)
        
        risk = update['risk_assessment']['risk_level']
        score = update['risk_assessment']['anomaly_score']
        
        print(f"  Step {i+1}: Risk={risk:8s}, Score={score:.3f}, Pos={position}")
        
        # Replan if needed
        if update['path_update_required']:
            engine.replan_trajectory(position, (50, 50, 50))
            print(f"    → Path replanned!")
    
    # Get summary
    status = engine.get_system_status()
    print(f"\nFlight Summary:")
    if 'risk_summary' in status:
        print(f"  HIGH risk events: {status['risk_summary']['high_count']}")
        print(f"  CRITICAL risk events: {status['risk_summary']['critical_count']}")
    
    engine.stop_mission()
    print()


def example_telemetry_features():
    """Example showing telemetry feature details"""
    print("=" * 60)
    print("Example 5: Telemetry Features")
    print("=" * 60)
    
    simulator = TelemetrySimulator()
    
    print("\nFeature Names:")
    for i, name in enumerate(simulator.get_feature_names()):
        print(f"  {i+1}. {name}")
    
    print("\nNormal Telemetry Sample:")
    normal = simulator.generate_normal_telemetry(1)[0]
    for i, (name, value) in enumerate(zip(simulator.get_feature_names(), normal)):
        print(f"  {name:20s}: {value:6.2f}")
    
    print("\nAnomalous Telemetry Sample (Battery):")
    anomalous = simulator.generate_anomalous_telemetry(1, 'battery')[0]
    for i, (name, value) in enumerate(zip(simulator.get_feature_names(), anomalous)):
        marker = " ⚠" if name == 'battery_level' and value < 50 else ""
        print(f"  {name:20s}: {value:6.2f}{marker}")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DIGITAL TWIN SYSTEM - EXAMPLES" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Run examples
    example_basic_usage()
    example_custom_anomaly_detector()
    example_path_planning()
    example_real_time_monitoring()
    example_telemetry_features()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
