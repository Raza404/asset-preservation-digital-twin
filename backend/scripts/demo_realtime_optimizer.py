"""
Real-time Flight Optimizer Demo
Demonstrates hybrid LSTM + RF system providing live recommendations.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.hybrid_decision_engine import HybridDecisionEngine


def simulate_telemetry_stream(scenario: str = 'normal'):
    """
    Simulate realistic telemetry data stream.
    
    Scenarios:
    - normal: Healthy flight
    - motor_degradation: Gradually failing motor
    - low_battery: Battery depleting
    - vibration: Propeller damage
    """
    base_telemetry = {
        'battery_voltage': 12.4,
        'battery_remaining': 85,
        'throttle': 55,
        'ground_speed': 12.0,
        'altitude': 50.0,
        'motor_temp': 55,
        'esc_temp': 50,
        'vibration_magnitude': 0.3,
        'gps_satellites': 14,
        'gyro_x': 10, 'gyro_y': -5, 'gyro_z': 2,
        'accel_x': 0.2, 'accel_y': -0.1, 'accel_z': 9.8
    }
    
    t = 0
    while True:
        telemetry = base_telemetry.copy()
        
        # Apply scenario
        if scenario == 'motor_degradation':
            # Motor gradually heats up and vibrates
            degradation_factor = min(1.0, t / 100)
            telemetry['motor_temp'] = 55 + (40 * degradation_factor)  # 55 -> 95°C
            telemetry['vibration_magnitude'] = 0.3 + (1.5 * degradation_factor)  # Increasing vibration
            telemetry['throttle'] = 55 + (10 * degradation_factor)  # Needs more throttle
        
        elif scenario == 'low_battery':
            # Battery drains
            drain_factor = min(1.0, t / 80)
            telemetry['battery_voltage'] = 12.4 - (2.0 * drain_factor)  # 12.4V -> 10.4V
            telemetry['battery_remaining'] = 85 - (80 * drain_factor)  # 85% -> 5%
        
        elif scenario == 'vibration':
            # Sudden vibration spike (damaged prop)
            if t > 20:
                telemetry['vibration_magnitude'] = 1.8 + np.random.uniform(-0.3, 0.3)
                telemetry['gyro_x'] = np.random.uniform(-50, 50)
                telemetry['gyro_y'] = np.random.uniform(-50, 50)
        
        # Add realistic noise
        telemetry['battery_voltage'] += np.random.uniform(-0.05, 0.05)
        telemetry['throttle'] += np.random.uniform(-2, 2)
        telemetry['ground_speed'] += np.random.uniform(-0.5, 0.5)
        
        t += 1
        yield telemetry


def print_colored(text: str, color: str = 'white'):
    """Print colored text (basic implementation)."""
    colors = {
        'red': '\033[91m',
        'yellow': '\033[93m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def format_health_bar(score: float, width: int = 20) -> str:
    """Create visual health bar."""
    filled = int(score * width)
    empty = width - filled
    
    if score > 0.7:
        color = '🟩'
    elif score > 0.4:
        color = '🟨'
    else:
        color = '🟥'
    
    bar = color * filled + '⬜' * empty
    return f"{bar} {score*100:.1f}%"


def demo_realtime_optimization(scenario: str = 'normal', duration: int = 30):
    """
    Run real-time optimization demo.
    
    Args:
        scenario: Flight scenario to simulate
        duration: How many seconds to run
    """
    print("\n" + "="*80)
    print("🚁 REAL-TIME FLIGHT OPTIMIZER - LIVE DEMO")
    print("="*80)
    print(f"\nScenario: {scenario.upper()}")
    print(f"Duration: {duration} seconds")
    print(f"Update Rate: 10 Hz\n")
    
    # Initialize decision engine
    print("Initializing hybrid decision engine...")
    rf_model_path = os.path.join('..', 'src', 'ml', 'failure_model.joblib')
    engine = HybridDecisionEngine(rf_model_path=rf_model_path)
    print("✓ Engine ready\n")
    
    # Start telemetry stream
    telemetry_stream = simulate_telemetry_stream(scenario)
    
    print("="*80)
    print("LIVE MONITORING (Press Ctrl+C to stop)")
    print("="*80 + "\n")
    
    try:
        for i in range(duration * 10):  # 10 Hz
            telemetry = next(telemetry_stream)
            
            # Process telemetry
            result = engine.process_telemetry(telemetry)
            
            # Display every 10th update (1 Hz display)
            if i % 10 == 0:
                _display_update(i // 10, result, telemetry)
            
            time.sleep(0.1)  # 10 Hz
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
    
    # Final summary
    print("\n" + "="*80)
    print("SESSION SUMMARY")
    print("="*80)
    print(engine.get_summary_report())


def _display_update(second: int, result: Dict, telemetry: Dict):
    """Display single update."""
    print(f"\n⏱️  T+{second:03d}s " + "-"*70)
    
    # Flight state
    print(f"📊 Flight State:")
    print(f"   Battery: {telemetry['battery_voltage']:.2f}V ({telemetry['battery_remaining']:.0f}%)")
    print(f"   Throttle: {telemetry['throttle']:.1f}% | Speed: {telemetry['ground_speed']:.1f} m/s")
    print(f"   Altitude: {telemetry['altitude']:.1f}m | GPS Sats: {telemetry['gps_satellites']}")
    
    # Risk level
    risk = result['fused_analysis']['overall_risk']
    risk_color = 'red' if risk > 0.7 else 'yellow' if risk > 0.5 else 'green'
    print(f"\n🎯 Overall Risk: ", end='')
    print_colored(f"{risk*100:.1f}%", risk_color)
    
    # Component health
    component_risks = result['fused_analysis'].get('component_risks', {})
    if component_risks:
        print(f"\n💊 Component Health:")
        for comp, info in list(component_risks.items())[:5]:  # Top 5
            health_score = 1.0 - info['current_risk']
            bar = format_health_bar(health_score, width=15)
            status = info['status']
            status_emoji = '✅' if status == 'HEALTHY' else '⚠️' if status == 'WARNING' else '🚨'
            print(f"   {status_emoji} {comp:12s}: {bar}")
    
    # Actions
    actions = result.get('recommended_actions', [])
    if actions:
        print(f"\n🎬 Recommended Actions ({len(actions)}):")
        for action in actions[:3]:  # Top 3
            print(f"   {action['display']}")
    else:
        print(f"\n✅ No actions needed - All systems nominal")
    
    # Immediate action required?
    if result.get('requires_immediate_action'):
        print_colored("\n⚠️⚠️⚠️  IMMEDIATE ACTION REQUIRED  ⚠️⚠️⚠️", 'red')


def main():
    """Main demo entry point."""
    print("\n🚁 HYBRID LSTM + RF FLIGHT OPTIMIZER")
    print("="*80)
    
    scenarios = {
        '1': ('normal', 'Normal healthy flight'),
        '2': ('motor_degradation', 'Motor gradually failing'),
        '3': ('low_battery', 'Battery depleting'),
        '4': ('vibration', 'Propeller damage with vibration')
    }
    
    print("\nAvailable Scenarios:")
    for key, (scenario, desc) in scenarios.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect scenario (1-4) [default: 1]: ").strip() or '1'
    scenario, desc = scenarios.get(choice, scenarios['1'])
    
    duration = input("Duration in seconds [default: 30]: ").strip()
    duration = int(duration) if duration.isdigit() else 30
    
    # Run demo
    demo_realtime_optimization(scenario, duration)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
