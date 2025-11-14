"""
Main demonstration script for the Digital Twin System
Shows real-time anomaly detection and adaptive path planning
"""

import numpy as np
from digital_twin_engine import DigitalTwinEngine
from telemetry_simulator import TelemetrySimulator


def main():
    """Main demonstration of the digital twin system"""
    
    print("=" * 70)
    print("DIGITAL TWIN SYSTEM - DRONE FAILURE PREDICTION & TRAJECTORY OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Initialize components
    print("1. SYSTEM INITIALIZATION")
    print("-" * 70)
    
    simulator = TelemetrySimulator(seed=42)
    engine = DigitalTwinEngine(contamination=0.1, safety_margin=10.0)
    
    # Generate training data
    print("Generating training data...")
    training_data = simulator.generate_training_dataset(num_samples=1000)
    print(f"✓ Generated {len(training_data)} training samples")
    
    # Initialize engine
    engine.initialize(training_data)
    print()
    
    # Start mission
    print("2. MISSION START")
    print("-" * 70)
    start_pos = (0.0, 0.0, 50.0)
    end_pos = (100.0, 80.0, 50.0)
    
    mission_info = engine.start_mission(start_pos, end_pos)
    print()
    
    # Simulate flight with anomaly detection
    print("3. REAL-TIME MONITORING & ADAPTIVE CONTROL")
    print("-" * 70)
    
    # Generate flight sequence
    telemetry_sequence, positions = simulator.simulate_flight_sequence(
        duration=20,
        anomaly_start=12
    )
    
    print("\nFlight Progress:")
    print()
    
    path_replanned = False
    
    for i, (telemetry, position) in enumerate(zip(telemetry_sequence, positions)):
        print(f"Time Step {i+1:2d} | Position: ({position[0]:6.1f}, {position[1]:6.1f}, {position[2]:5.1f})")
        
        # Update telemetry
        update = engine.update_telemetry(telemetry, position)
        risk = update['risk_assessment']
        
        print(f"            | Risk: {risk['risk_level']:8s} | Anomaly Score: {risk['anomaly_score']:.3f}")
        
        # Replan if needed and not already replanned
        if update['path_update_required'] and not path_replanned:
            print(f"            | ⚠ PATH REPLANNING TRIGGERED")
            replan_result = engine.replan_trajectory(position, end_pos)
            path_replanned = True
            print(f"            | ✓ New path: {replan_result['path_metrics']['num_waypoints']} waypoints")
        
        # Show recommendation for high-risk situations
        if risk['risk_level'] in ['HIGH', 'CRITICAL']:
            print(f"            | → {risk['recommendation']}")
        
        print()
    
    # Mission summary
    print("4. MISSION SUMMARY")
    print("-" * 70)
    
    summary = engine.stop_mission()
    status = engine.get_system_status()
    
    print(f"Total telemetry updates: {summary['total_telemetry_updates']}")
    print(f"Final position: {summary['final_position']}")
    print()
    
    if 'risk_summary' in status:
        risk_summary = status['risk_summary']
        print("Risk Level Distribution:")
        print(f"  LOW:      {risk_summary['low_count']}")
        print(f"  MEDIUM:   {risk_summary['medium_count']}")
        print(f"  HIGH:     {risk_summary['high_count']}")
        print(f"  CRITICAL: {risk_summary['critical_count']}")
        print()
        print(f"Average Anomaly Score: {risk_summary['avg_anomaly_score']:.3f}")
        print(f"Maximum Anomaly Score: {risk_summary['max_anomaly_score']:.3f}")
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
