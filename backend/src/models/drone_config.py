from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path


class FrameType(Enum):
    """Drone frame types"""
    X = "X"
    PLUS = "Plus"
    H = "H"
    CUSTOM = "Custom"


class MaterialType(Enum):
    """Frame material types"""
    CARBON_FIBER = "Carbon Fiber"
    ALUMINUM = "Aluminum"
    PLASTIC = "Plastic"
    COMPOSITE = "Composite"


@dataclass
class MotorConfig:
    """Motor configuration"""
    motor_id: int
    max_thrust_n: float  # Newtons
    max_rpm: int
    weight_kg: float
    expected_lifetime_hours: float
    critical_temp_celsius: float


@dataclass
class ArmConfig:
    """Arm configuration"""
    arm_id: int
    length_m: float
    cross_section_area_m2: float
    material: MaterialType
    thickness_mm: float
    max_bending_stress_mpa: float  # Megapascals
    fatigue_limit_mpa: float
    motor_id: int  # Which motor is attached to this arm


@dataclass
class BatteryConfig:
    """Battery configuration"""
    capacity_mah: int
    voltage_nominal: float
    voltage_max: float
    voltage_min: float
    chemistry: str  # LiPo, Li-ion, etc.
    cells: int  # Number of cells (S rating)
    weight_kg: float
    max_discharge_c: float
    max_charge_c: float
    expected_cycle_life: int


@dataclass
class DroneConfig:
    """Complete drone configuration"""
    
    # Identification
    drone_id: str
    model_name: str
    
    # Frame
    frame_type: FrameType
    total_weight_kg: float
    frame_material: MaterialType
    wheelbase_m: float  # Distance between opposite motors
    
    # Components
    motors: List[MotorConfig] = field(default_factory=list)
    arms: List[ArmConfig] = field(default_factory=list)
    battery: Optional[BatteryConfig] = None
    
    # Aerodynamics
    drag_coefficient: float = 0.6
    frontal_area_m2: float = 0.1
    
    # Performance limits
    max_speed_ms: float = 15.0
    max_climb_rate_ms: float = 5.0
    max_bank_angle_deg: float = 45.0
    max_yaw_rate_degs: float = 180.0
    max_wind_speed_ms: float = 10.0
    
    @property
    def num_motors(self) -> int:
        """Get number of motors"""
        return len(self.motors)
    
    @property
    def total_max_thrust_n(self) -> float:
        """Calculate total maximum thrust"""
        return sum(motor.max_thrust_n for motor in self.motors)
    
    @property
    def thrust_to_weight_ratio(self) -> float:
        """Calculate thrust-to-weight ratio"""
        weight_n = self.total_weight_kg * 9.81
        return self.total_max_thrust_n / weight_n if weight_n > 0 else 0
    
    @property
    def hover_thrust_per_motor_n(self) -> float:
        """Calculate hover thrust per motor"""
        if self.num_motors == 0:
            return 0
        weight_n = self.total_weight_kg * 9.81
        return weight_n / self.num_motors
    
    @property
    def hover_throttle_percent(self) -> float:
        """Calculate hover throttle percentage"""
        if self.total_max_thrust_n == 0:
            return 0
        weight_n = self.total_weight_kg * 9.81
        return (weight_n / self.total_max_thrust_n) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'drone_id': self.drone_id,
            'model_name': self.model_name,
            'frame_type': self.frame_type.value,
            'total_weight_kg': self.total_weight_kg,
            'frame_material': self.frame_material.value,
            'wheelbase_m': self.wheelbase_m,
            'motors': [
                {
                    'motor_id': m.motor_id,
                    'max_thrust_n': m.max_thrust_n,
                    'max_rpm': m.max_rpm,
                    'weight_kg': m.weight_kg,
                    'expected_lifetime_hours': m.expected_lifetime_hours,
                    'critical_temp_celsius': m.critical_temp_celsius
                }
                for m in self.motors
            ],
            'arms': [
                {
                    'arm_id': a.arm_id,
                    'length_m': a.length_m,
                    'cross_section_area_m2': a.cross_section_area_m2,
                    'material': a.material.value,
                    'thickness_mm': a.thickness_mm,
                    'max_bending_stress_mpa': a.max_bending_stress_mpa,
                    'fatigue_limit_mpa': a.fatigue_limit_mpa,
                    'motor_id': a.motor_id
                }
                for a in self.arms
            ],
            'battery': {
                'capacity_mah': self.battery.capacity_mah,
                'voltage_nominal': self.battery.voltage_nominal,
                'voltage_max': self.battery.voltage_max,
                'voltage_min': self.battery.voltage_min,
                'chemistry': self.battery.chemistry,
                'cells': self.battery.cells,
                'weight_kg': self.battery.weight_kg,
                'max_discharge_c': self.battery.max_discharge_c,
                'max_charge_c': self.battery.max_charge_c,
                'expected_cycle_life': self.battery.expected_cycle_life
            } if self.battery else None,
            'drag_coefficient': self.drag_coefficient,
            'frontal_area_m2': self.frontal_area_m2,
            'max_speed_ms': self.max_speed_ms,
            'max_climb_rate_ms': self.max_climb_rate_ms,
            'max_bank_angle_deg': self.max_bank_angle_deg,
            'max_yaw_rate_degs': self.max_yaw_rate_degs,
            'max_wind_speed_ms': self.max_wind_speed_ms
        }
    
    @staticmethod
    def _parse_frame_type(frame_str: str) -> FrameType:
        """Parse frame type string (handles multiple formats)"""
        frame_map = {
            'x': FrameType.X,
            'X': FrameType.X,
            'plus': FrameType.PLUS,
            'Plus': FrameType.PLUS,
            'PLUS': FrameType.PLUS,
            'h': FrameType.H,
            'H': FrameType.H,
            'custom': FrameType.CUSTOM,
            'Custom': FrameType.CUSTOM,
            'CUSTOM': FrameType.CUSTOM,
        }
        return frame_map.get(frame_str, FrameType.X)
    
    @staticmethod
    def _parse_material(material_str: str) -> MaterialType:
        """Parse material string (handles multiple formats)"""
        material_map = {
            'carbon_fiber': MaterialType.CARBON_FIBER,
            'Carbon Fiber': MaterialType.CARBON_FIBER,
            'CARBON_FIBER': MaterialType.CARBON_FIBER,
            'aluminum': MaterialType.ALUMINUM,
            'Aluminum': MaterialType.ALUMINUM,
            'ALUMINUM': MaterialType.ALUMINUM,
            'plastic': MaterialType.PLASTIC,
            'Plastic': MaterialType.PLASTIC,
            'PLASTIC': MaterialType.PLASTIC,
            'composite': MaterialType.COMPOSITE,
            'Composite': MaterialType.COMPOSITE,
            'COMPOSITE': MaterialType.COMPOSITE,
        }
        return material_map.get(material_str, MaterialType.CARBON_FIBER)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DroneConfig':
        """Create from dictionary"""
        
        # Parse motors
        motors = [
            MotorConfig(
                motor_id=m['motor_id'],
                max_thrust_n=m['max_thrust_n'],
                max_rpm=m['max_rpm'],
                weight_kg=m['weight_kg'],
                expected_lifetime_hours=m['expected_lifetime_hours'],
                critical_temp_celsius=m['critical_temp_celsius']
            )
            for m in data.get('motors', [])
        ]
        
        # Parse arms with material type conversion
        arms = [
            ArmConfig(
                arm_id=a['arm_id'],
                length_m=a['length_m'],
                cross_section_area_m2=a['cross_section_area_m2'],
                material=cls._parse_material(a['material']),
                thickness_mm=a['thickness_mm'],
                max_bending_stress_mpa=a['max_bending_stress_mpa'],
                fatigue_limit_mpa=a['fatigue_limit_mpa'],
                motor_id=a['motor_id']
            )
            for a in data.get('arms', [])
        ]
        
        # Parse battery
        battery = None
        if data.get('battery'):
            b = data['battery']
            battery = BatteryConfig(
                capacity_mah=b['capacity_mah'],
                voltage_nominal=b['voltage_nominal'],
                voltage_max=b['voltage_max'],
                voltage_min=b['voltage_min'],
                chemistry=b['chemistry'],
                cells=b['cells'],
                weight_kg=b['weight_kg'],
                max_discharge_c=b['max_discharge_c'],
                max_charge_c=b['max_charge_c'],
                expected_cycle_life=b['expected_cycle_life']
            )
        
        return cls(
            drone_id=data['drone_id'],
            model_name=data['model_name'],
            frame_type=cls._parse_frame_type(data['frame_type']),
            total_weight_kg=data['total_weight_kg'],
            frame_material=cls._parse_material(data['frame_material']),
            wheelbase_m=data['wheelbase_m'],
            motors=motors,
            arms=arms,
            battery=battery,
            drag_coefficient=data.get('drag_coefficient', 0.6),
            frontal_area_m2=data.get('frontal_area_m2', 0.1),
            max_speed_ms=data.get('max_speed_ms', 15.0),
            max_climb_rate_ms=data.get('max_climb_rate_ms', 5.0),
            max_bank_angle_deg=data.get('max_bank_angle_deg', 45.0),
            max_yaw_rate_degs=data.get('max_yaw_rate_degs', 180.0),
            max_wind_speed_ms=data.get('max_wind_speed_ms', 10.0)
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DroneConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)