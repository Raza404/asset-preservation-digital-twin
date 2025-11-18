"""
Quick test of the hybrid system with simulated telemetry.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.hybrid_decision_engine import HybridDecisionEngine


def test_hybrid_system():
    """Test the hybrid decision engine."""
    print("\n" + "="*80)
    print("🧪 TESTING HYBRID LSTM + RF SYSTEM")
    print("="*80 + "\n")
    
    # Initialize engine (without pre-trained models for now)
    print("1️⃣ Initializing hybrid decision engine...")
    try:
        engine = HybridDecisionEngine()
        print("✅ Engine initialized\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        return
    
    # Test scenarios
    scenarios = {
        'healthy': {
            'battery_voltage': 12.4,
            'battery_remaining': 85,
            'throttle': 55,
            'motor_temp': 60,
            'vibration_magnitude': 0.3,
            'gps_satellites': 14
        },
        'low_battery': {
            'battery_voltage': 10.8,
            'battery_remaining': 15,
            'throttle': 65,
            'motor_temp': 65,
            'vibration_magnitude': 0.4,
            'gps_satellites': 12
        },
        'motor_issue': {
            'battery_voltage': 11.9,
            'battery_remaining': 55,
            'throttle': 75,
            'motor_temp': 95,
            'vibration_magnitude': 1.6,
            'gps_satellites': 13
        }
    }
    
    print("2️⃣ Testing scenarios...\n")
    
    for scenario_name, telemetry in scenarios.items():
        print(f"\n{'─'*80}")
        print(f"📊 Scenario: {scenario_name.upper()}")
        print(f"{'─'*80}")
        
        # Add required fields
        telemetry.update({
            'ground_speed': 12.0,
            'altitude': 50.0,
            'esc_temp': 55,
            'gyro_x': 10, 'gyro_y': -5, 'gyro_z': 2,
            'accel_x': 0.2, 'accel_y': -0.1, 'accel_z': 9.8
        })
        
        try:
            # Process telemetry
            result = engine.process_telemetry(telemetry)
            
            # Display results
            print(f"\n📈 Telemetry:")
            print(f"   Battery: {telemetry['battery_voltage']:.2f}V ({telemetry['battery_remaining']}%)")
            print(f"   Motor Temp: {telemetry['motor_temp']}°C")
            print(f"   Vibration: {telemetry['vibration_magnitude']:.2f}")
            print(f"   Throttle: {telemetry['throttle']}%")
            
            print(f"\n🎯 Overall Risk: {result['risk_level']*100:.1f}%")
            
            # Actions
            actions = result.get('recommended_actions', [])
            if actions:
                print(f"\n🎬 Recommended Actions ({len(actions)}):")
                for action in actions[:5]:
                    print(f"   {action['display']}")
            else:
                print(f"\n✅ No actions needed - All systems nominal")
            
            # Component health
            if 'fused_analysis' in result and 'component_risks' in result['fused_analysis']:
                print(f"\n💊 Component Status:")
                for comp, info in list(result['fused_analysis']['component_risks'].items())[:5]:
                    health = (1.0 - info['current_risk']) * 100
                    status = info['status']
                    emoji = '✅' if status == 'HEALTHY' else '⚠️' if status == 'WARNING' else '🚨'
                    print(f"   {emoji} {comp:12s}: {health:5.1f}% - {status}")
            
        except Exception as e:
            print(f"❌ Error processing scenario: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print("\nNote: System is running in fallback mode without pre-trained models.")
    print("For full functionality, train models using train_hybrid_models.py\n")


if __name__ == "__main__":
    test_hybrid_system()
