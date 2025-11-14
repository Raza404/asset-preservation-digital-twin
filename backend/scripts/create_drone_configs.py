import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.drone_config import (
    DroneConfig, MotorConfig, ArmConfig, BatteryConfig,
    FrameType, MaterialType
)
import math

def create_standard_quadcopter() -> DroneConfig:
    """Create a standard quadcopter (X-frame) configuration"""
    
    arm_length = 0.225  # 225mm
    angle_45 = math.radians(45)
    
    motors = [
        MotorConfig(
            motor_id=1,
            position_x=arm_length * math.cos(angle_45),
            position_y=arm_length * math.sin(angle_45),
            position_z=0.0,
            rotation_direction="CW",
            max_thrust_n=20.0,
            max_rpm=8000,
            weight_kg=0.055,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ),
        MotorConfig(
            motor_id=2,
            position_x=arm_length * math.cos(angle_45),
            position_y=-arm_length * math.sin(angle_45),
            position_z=0.0,
            rotation_direction="CCW",
            max_thrust_n=20.0,
            max_rpm=8000,
            weight_kg=0.055,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ),
        MotorConfig(
            motor_id=3,
            position_x=-arm_length * math.cos(angle_45),
            position_y=-arm_length * math.sin(angle_45),
            position_z=0.0,
            rotation_direction="CW",
            max_thrust_n=20.0,
            max_rpm=8000,
            weight_kg=0.055,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ),
        MotorConfig(
            motor_id=4,
            position_x=-arm_length * math.cos(angle_45),
            position_y=arm_length * math.sin(angle_45),
            position_z=0.0,
            rotation_direction="CCW",
            max_thrust_n=20.0,
            max_rpm=8000,
            weight_kg=0.055,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ),
    ]
    
    arms = [
        ArmConfig(
            arm_id=i+1,
            length_m=arm_length,
            cross_section_area_m2=0.0001,
            material=MaterialType.CARBON_FIBER,
            thickness_mm=2.0,
            max_bending_stress_mpa=600.0,
            fatigue_limit_mpa=300.0,
            motor_id=i+1
        ) for i in range(4)
    ]
    
    battery = BatteryConfig(
        capacity_mah=5000,
        voltage_nominal=14.8,
        voltage_max=16.8,
        voltage_min=12.8,
        chemistry="LiPo",
        cells=4,
        weight_kg=0.450,
        max_discharge_c=50.0,
        max_charge_c=5.0,
        expected_cycle_life=300
    )
    
    return DroneConfig(
        drone_id="QUAD_450_01",
        model_name="Standard 450mm Quadcopter",
        frame_type=FrameType.QUAD_X,
        total_weight_kg=1.5,
        frame_material=MaterialType.CARBON_FIBER,
        wheelbase_m=0.450,
        motors=motors,
        arms=arms,
        battery=battery,
        drag_coefficient=0.6,
        frontal_area_m2=0.09,
        max_speed_ms=15.0,
        max_climb_rate_ms=5.0,
        max_bank_angle_deg=45.0,
        max_yaw_rate_degs=180.0,
        max_wind_speed_ms=8.0
    )

def create_hexacopter() -> DroneConfig:
    """Create a hexacopter (6 motors) configuration"""
    
    arm_length = 0.300
    
    motors = []
    for i in range(6):
        angle = math.radians(i * 60)
        motors.append(MotorConfig(
            motor_id=i+1,
            position_x=arm_length * math.cos(angle),
            position_y=arm_length * math.sin(angle),
            position_z=0.0,
            rotation_direction="CW" if i % 2 == 0 else "CCW",
            max_thrust_n=18.0,
            max_rpm=7500,
            weight_kg=0.060,
            expected_lifetime_hours=200.0,
            critical_temp_celsius=85.0
        ))
    
    arms = [
        ArmConfig(
            arm_id=i+1,
            length_m=arm_length,
            cross_section_area_m2=0.00012,
            material=MaterialType.CARBON_FIBER,
            thickness_mm=2.5,
            max_bending_stress_mpa=600.0,
            fatigue_limit_mpa=300.0,
            motor_id=i+1
        ) for i in range(6)
    ]
    
    battery = BatteryConfig(
        capacity_mah=8000,
        voltage_nominal=22.2,
        voltage_max=25.2,
        voltage_min=19.2,
        chemistry="LiPo",
        cells=6,
        weight_kg=0.750,
        max_discharge_c=40.0,
        max_charge_c=5.0,
        expected_cycle_life=300
    )
    
    return DroneConfig(
        drone_id="HEX_600_01",
        model_name="600mm Hexacopter",
        frame_type=FrameType.HEX_X,
        total_weight_kg=2.5,
        frame_material=MaterialType.CARBON_FIBER,
        wheelbase_m=0.600,
        motors=motors,
        arms=arms,
        battery=battery,
        drag_coefficient=0.7,
        frontal_area_m2=0.15,
        max_speed_ms=12.0,
        max_climb_rate_ms=4.0,
        max_bank_angle_deg=35.0,
        max_yaw_rate_degs=120.0,
        max_wind_speed_ms=10.0
    )

if __name__ == "__main__":
    # Create output directory
    config_dir = os.path.join('..', 'config', 'drones')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create configurations
    quad = create_standard_quadcopter()
    quad.save_to_file(os.path.join(config_dir, 'quadcopter_450.json'))
    
    hex_drone = create_hexacopter()
    hex_drone.save_to_file(os.path.join(config_dir, 'hexacopter_600.json'))
    
    # Print summary
    print("\n" + "="*60)
    print("DRONE CONFIGURATIONS CREATED")
    print("="*60)
    
    for drone in [quad, hex_drone]:
        print(f"\n{drone.model_name}:")
        print(f"  Motors: {drone.num_motors}")
        print(f"  Total thrust: {drone.total_max_thrust_n:.1f} N")
        print(f"  Thrust-to-weight: {drone.thrust_to_weight_ratio:.2f}")
        print(f"  Weight: {drone.total_weight_kg} kg")
        print(f"  Wheelbase: {drone.wheelbase_m*1000} mm")
