"""
Demo: Production-Ready Rule-Based Flight Monitoring
Showcases sophisticated failure prediction without ML models.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.rule_based_predictor import RuleBasedPredictor


def generate_telemetry_scenario(scenario: str, timesteps: int = 30) -> list:
    """Generate telemetry for different flight scenarios."""
    telemetry_buffer = []
    
    for i in range(timesteps):
        if scenario == "healthy":
            telemetry = {
                'battery_voltage': 12.4 - (i * 0.01),
                'battery_remaining': 85 - i,
                'throttle': 55 + np.random.uniform(-5, 5),
                'ground_speed': 12.0,
                'altitude': 50.0 + np.random.uniform(-2, 2),
                'motor_temp': 60 + i * 0.2,
                'esc_temp': 50 + i * 0.15,
                'vibration_magnitude': 0.3 + np.random.uniform(-0.1, 0.1),
                'gps_satellites': 14,
                'gyro_x': np.random.uniform(-10, 10),
                'gyro_y': np.random.uniform(-10, 10),
                'gyro_z': np.random.uniform(-5, 5)
            }
        
        elif scenario == "battery_critical":
            # Simulating rapid battery drain
            telemetry = {
                'battery_voltage': 11.5 - (i * 0.08),  # Fast discharge
                'battery_remaining': 25 - (i * 2),
                'throttle': 75,  # High throttle
                'ground_speed': 15.0,
                'altitude': 50.0,
                'motor_temp': 70,
                'esc_temp': 60,
                'vibration_magnitude': 0.4,
                'gps_satellites': 12,
                'gyro_x': np.random.uniform(-10, 10),
                'gyro_y': np.random.uniform(-10, 10),
                'gyro_z': np.random.uniform(-5, 5)
            }
        
        elif scenario == "motor_overheat":
            # Motor overheating scenario
            telemetry = {
                'battery_voltage': 12.2,
                'battery_remaining': 70,
                'throttle': 85,  # High throttle
                'ground_speed': 18.0,
                'altitude': 60.0,
                'motor_temp': 65 + (i * 1.2),  # Rapid heating
                'esc_temp': 55 + (i * 0.8),
                'vibration_magnitude': 0.6 + (i * 0.03),  # Increasing vibration
                'gps_satellites': 13,
                'gyro_x': np.random.uniform(-15, 15),
                'gyro_y': np.random.uniform(-15, 15),
                'gyro_z': np.random.uniform(-8, 8)
            }
        
        elif scenario == "propeller_damage":
            # Damaged propeller causing severe vibration
            telemetry = {
                'battery_voltage': 12.3,
                'battery_remaining': 75,
                'throttle': 70 + np.random.uniform(-10, 10),
                'ground_speed': 10.0,  # Low speed despite high throttle
                'altitude': 45.0,  # Poor climb performance
                'motor_temp': 75,  # Working hard
                'esc_temp': 65,
                'vibration_magnitude': 1.8 + np.random.uniform(-0.2, 0.4),  # Very high
                'gps_satellites': 12,
                'gyro_x': np.random.uniform(-30, 30),  # High noise
                'gyro_y': np.random.uniform(-30, 30),
                'gyro_z': np.random.uniform(-20, 20)
            }
        
        elif scenario == "gps_degradation":
            # GPS signal loss
            telemetry = {
                'battery_voltage': 12.1,
                'battery_remaining': 60,
                'throttle': 60,
                'ground_speed': 12.0,
                'altitude': 55.0,
                'motor_temp': 65,
                'esc_temp': 55,
                'vibration_magnitude': 0.35,
                'gps_satellites': max(4, 12 - i),  # Degrading GPS
                'gyro_x': np.random.uniform(-50, 50),  # High IMU noise
                'gyro_y': np.random.uniform(-50, 50),
                'gyro_z': np.random.uniform(-30, 30)
            }
        
        else:  # combined_stress
            # Multiple issues
            telemetry = {
                'battery_voltage': 11.2 - (i * 0.05),
                'battery_remaining': 30 - i,
                'throttle': 80,
                'ground_speed': 16.0,
                'altitude': 70.0,
                'motor_temp': 80 + (i * 0.5),
                'esc_temp': 70 + (i * 0.4),
                'vibration_magnitude': 1.2 + (i * 0.02),
                'gps_satellites': max(6, 11 - (i // 5)),
                'gyro_x': np.random.uniform(-25, 25),
                'gyro_y': np.random.uniform(-25, 25),
                'gyro_z': np.random.uniform(-15, 15)
            }
        
        telemetry_buffer.append(telemetry)
    
    return telemetry_buffer


def print_component_analysis(comp_name: str, health):
    """Pretty print component health analysis."""
    status_emoji = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'CRITICAL': '🚨'
    }
    
    emoji = status_emoji.get(health.status, '❓')
    
    print(f"\n{'='*60}")
    print(f"{emoji} {comp_name.upper()} - {health.status}")
    print(f"{'='*60}")
    print(f"Risk Level: {health.risk_level:.1%}")
    print(f"Remaining Useful Life: {health.predicted_rul}s ({health.predicted_rul//60}m {health.predicted_rul%60}s)")
    print(f"\nTriggered Alerts:")
    for trigger in health.triggers:
        print(f"  • {trigger.replace('_', ' ').title()}")
    
    print(f"\nDiagnostic Info:")
    for key, value in health.diagnostic_info.items():
        if key != 'triggers':
            if isinstance(value, float):
                print(f"  • {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(health.recommendations, 1):
        print(f"  {i}. {rec}")


def main():
    print("="*70)
    print("PRODUCTION-READY RULE-BASED FLIGHT MONITORING SYSTEM")
    print("Sophisticated Failure Prediction Without ML Models")
    print("="*70)
    
    predictor = RuleBasedPredictor()
    
    scenarios = [
        ("healthy", "Normal flight conditions"),
        ("battery_critical", "Rapid battery drainage"),
        ("motor_overheat", "Motor overheating under load"),
        ("propeller_damage", "Damaged propeller - severe vibration"),
        ("gps_degradation", "GPS signal degradation"),
        ("combined_stress", "Multiple simultaneous failures")
    ]
    
    for scenario_name, description in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"SCENARIO: {description.upper()}")
        print(f"{'#'*70}")
        
        # Generate telemetry
        telemetry_buffer = generate_telemetry_scenario(scenario_name, timesteps=30)
        
        # Analyze
        print("\n⏳ Analyzing 30 seconds of telemetry...")
        start_time = time.time()
        
        results = predictor.analyze_telemetry(telemetry_buffer)
        
        analysis_time = (time.time() - start_time) * 1000
        print(f"✓ Analysis complete in {analysis_time:.1f}ms")
        
        # Display results for each component
        for comp_name, health in results.items():
            print_component_analysis(comp_name, health)
        
        # Overall risk assessment
        print(f"\n{'='*60}")
        risks = [health.risk_level for health in results.values()]
        overall_risk = max(risks) if risks else 0.0
        critical_components = [name for name, health in results.items() if health.status == 'CRITICAL']
        warning_components = [name for name, health in results.items() if health.status == 'WARNING']
        
        print(f"📊 OVERALL FLIGHT RISK: {overall_risk:.1%}")
        
        if critical_components:
            print(f"🚨 CRITICAL COMPONENTS: {', '.join(critical_components)}")
            print(f"🚨 ACTION REQUIRED: Immediate landing recommended")
        elif warning_components:
            print(f"⚠️  WARNING COMPONENTS: {', '.join(warning_components)}")
            print(f"⚠️  ACTION REQUIRED: Reduce workload and plan landing")
        else:
            print(f"✅ All systems nominal - continue normal operation")
        
        print(f"{'='*60}")
        
        # Pause between scenarios
        if scenario_name != scenarios[-1][0]:
            print("\n" + "."*70)
            time.sleep(1)
    
    print(f"\n\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}")
    print("\n📝 Key Takeaways:")
    print("  ✓ Real-time component health monitoring without ML models")
    print("  ✓ Sophisticated heuristics detect:")
    print("    - Battery voltage curves and discharge rates")
    print("    - Motor thermal trends and vibration patterns")
    print("    - ESC temperature and efficiency metrics")
    print("    - Sensor quality and GPS degradation")
    print("    - Propeller damage through vibration analysis")
    print("  ✓ Actionable recommendations for each failure mode")
    print("  ✓ Sub-second response time for critical alerts")
    print("\n🚀 Production-ready for immediate deployment!")


if __name__ == "__main__":
    main()
