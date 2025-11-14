import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from src.models.drone_config import (
    DroneConfig, FrameType, MaterialType,
    MotorConfig, ArmConfig, BatteryConfig
)

def create_my_betaflight_drone():
    """Create configuration for my Betaflight drone"""
    
    print("="*60)
    print("🔧 CREATING YOUR DRONE CONFIGURATION")
    print("="*60)
    
    # From your Betaflight log analysis
    print("\nBased on your flight log:")
    print("  - 4 motors (quadcopter)")
    print("  - Battery: ~24.87V (6S LiPo)")
    print("  - GPS enabled")
    print("  - ~114 second flight")
    
    # Let's estimate based on common specs
    print("\n📝 Please provide some basic specs:")
    
    # Get user input
    wheelbase = input("\nWheelbase in mm (distance between opposite motors, e.g., 450): ").strip()
    wheelbase_m = float(wheelbase) / 1000 if wheelbase else 0.45
    
    total_weight = input("Total weight in kg (e.g., 1.5): ").strip()
    total_weight_kg = float(total_weight) if total_weight else 1.5
    
    motor_thrust = input("Max thrust per motor in kg (e.g., 1.2): ").strip()
    motor_thrust_kg = float(motor_thrust) if motor_thrust else 1.2
    motor_thrust_n = motor_thrust_kg * 9.81  # Convert to Newtons
    
    # Motors
    motors = []
    for i in range(4):
        motors.append(MotorConfig(
            motor_id=i+1,
            max_thrust_n=motor_thrust_n,
            max_rpm=8000,
            weight_kg=0.08,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ))
    
    # Arms
    arms = []
    arm_length = wheelbase_m / 2
    for i in range(4):
        arms.append(ArmConfig(
            arm_id=i+1,
            length_m=arm_length,
            cross_section_area_m2=0.000004,
            material=MaterialType.CARBON_FIBER,
            thickness_mm=2.5,
            max_bending_stress_mpa=600.0,
            fatigue_limit_mpa=300.0,
            motor_id=i+1
        ))
    
    # Battery (6S from log)
    battery = BatteryConfig(
        capacity_mah=5000,
        voltage_nominal=22.2,  # 6S
        voltage_max=25.2,
        voltage_min=19.2,
        chemistry="LiPo",
        cells=6,
        weight_kg=0.65,
        max_discharge_c=50.0,
        max_charge_c=5.0,
        expected_cycle_life=300
    )
    
    # Complete config
    config = DroneConfig(
        drone_id="MY_BETAFLIGHT_DRONE",
        model_name="My Betaflight Quadcopter",
        frame_type=FrameType.X,
        total_weight_kg=total_weight_kg,
        frame_material=MaterialType.CARBON_FIBER,
        wheelbase_m=wheelbase_m,
        motors=motors,
        arms=arms,
        battery=battery,
        drag_coefficient=0.6,
        frontal_area_m2=0.08,
        max_speed_ms=20.0,
        max_climb_rate_ms=8.0,
        max_bank_angle_deg=60.0,
        max_yaw_rate_degs=200.0,
        max_wind_speed_ms=10.0
    )
    
    # Save
    output_path = Path(__file__).parent.parent.parent / 'config' / 'drones' / 'my_betaflight_drone.json'
    config.save_to_file(str(output_path))
    
    print(f"\n✅ Saved: {output_path}")
    print(f"\n📊 Your Drone Stats:")
    print(f"   Total thrust: {config.total_max_thrust_n:.1f}N ({config.total_max_thrust_n/9.81:.1f}kg)")
    print(f"   Weight: {config.total_weight_kg}kg")
    print(f"   Thrust-to-Weight: {config.thrust_to_weight_ratio:.2f}")
    print(f"   Hover throttle: {config.hover_throttle_percent:.1f}%")
    
    return config

if __name__ == "__main__":
    create_my_betaflight_drone()