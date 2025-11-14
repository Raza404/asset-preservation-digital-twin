import numpy as np
from typing import Dict
from src.models.drone_config import DroneConfig

class StressCalculator:
    """Calculate structural stress on drone components"""
    
    def __init__(self, config: DroneConfig):
        self.config = config
    
    def calculate_flight_stress(self, 
                               g_force: float = 1.0,
                               wind_speed: float = 0.0,
                               air_temperature: float = 25.0,
                               altitude: float = 0.0) -> Dict:
        """Calculate overall flight stress score"""
        
        # Base stress from G-force
        g_stress = self._calculate_g_stress(g_force)
        
        # Wind stress
        wind_stress = self._calculate_wind_stress(wind_speed)
        
        # Temperature stress
        temp_stress = self._calculate_temperature_stress(air_temperature)
        
        # Altitude stress (air density effects)
        altitude_stress = self._calculate_altitude_stress(altitude)
        
        # Calculate component stresses
        motor_stress = self._calculate_motor_stress(g_force, air_temperature)
        arm_stress = self._calculate_arm_stress(g_force)
        battery_stress = self._calculate_battery_stress(air_temperature)
        
        # Overall stress (weighted average)
        overall_stress = (
            g_stress * 0.4 +
            wind_stress * 0.2 +
            temp_stress * 0.15 +
            motor_stress * 0.15 +
            arm_stress * 0.1
        )
        
        return {
            'overall_stress': min(100, max(0, overall_stress)),
            'g_stress': g_stress,
            'wind_stress': wind_stress,
            'temperature_stress': temp_stress,
            'altitude_stress': altitude_stress,
            'motor_stress': motor_stress,
            'arm_stress': arm_stress,
            'battery_stress': battery_stress,
            'components': {
                'motors': motor_stress,
                'arms': arm_stress,
                'battery': battery_stress
            }
        }
    
    def _calculate_g_stress(self, g_force: float) -> float:
        """Calculate stress from G-force"""
        
        if g_force <= 1.0:
            return 0.0
        
        # Exponential increase in stress
        # 2G = 20%, 3G = 45%, 4G = 70%, 5G+ = 95%
        normalized = (g_force - 1.0) / 4.0  # Normalize to 0-1 for 1-5G
        stress = 100 * (1 - np.exp(-3 * normalized))
        
        return min(100, stress)
    
    def _calculate_wind_stress(self, wind_speed: float) -> float:
        """Calculate stress from wind"""
        
        if wind_speed <= 0:
            return 0.0
        
        # Stress increases with wind speed
        # Normalized against max_wind_speed
        wind_ratio = wind_speed / self.config.max_wind_speed_ms
        
        if wind_ratio < 0.5:
            return wind_ratio * 40  # 0-20% stress for light wind
        elif wind_ratio < 0.8:
            return 20 + (wind_ratio - 0.5) * 100  # 20-50% for moderate
        else:
            return 50 + (wind_ratio - 0.8) * 250  # 50-100% for strong
    
    def _calculate_temperature_stress(self, temperature: float) -> float:
        """Calculate stress from temperature"""
        
        optimal_temp = 25.0  # Celsius
        
        if 15 <= temperature <= 35:
            return 0.0  # Optimal range
        
        # Stress increases outside optimal range
        if temperature < 15:
            # Cold stress
            stress = (15 - temperature) * 3
        else:
            # Heat stress
            stress = (temperature - 35) * 4
        
        return min(100, max(0, stress))
    
    def _calculate_altitude_stress(self, altitude: float) -> float:
        """Calculate stress from altitude (air density effects)"""
        
        if altitude < 1000:
            return 0.0
        
        # Simplified air density model
        # Stress increases with altitude due to thinner air
        altitude_km = altitude / 1000.0
        stress = altitude_km * 5  # 5% per 1000m
        
        return min(50, stress)
    
    def _calculate_motor_stress(self, g_force: float, temperature: float) -> float:
        """Calculate motor stress"""
        
        # Motors work harder with higher G-forces
        base_load = self.config.hover_throttle_percent / 100.0
        
        # G-force increases motor load
        g_load = g_force * base_load
        
        # Calculate throttle percentage needed
        throttle = min(100, g_load * 100)
        
        # Temperature affects motor efficiency
        temp_factor = 1.0
        if temperature > 30:
            temp_factor = 1 + (temperature - 30) * 0.02
        elif temperature < 10:
            temp_factor = 1 + (10 - temperature) * 0.01
        
        motor_stress = throttle * temp_factor * 0.8  # Scale to reasonable range
        
        return min(100, motor_stress)
    
    def _calculate_arm_stress(self, g_force: float) -> float:
        """Calculate arm/frame stress from G-forces"""
        
        if not self.config.arms:
            return 0.0
        
        # Calculate bending stress on arms
        # Stress = Force * Length / Cross_section
        
        weight_force = self.config.total_weight_kg * 9.81 * g_force
        force_per_arm = weight_force / self.config.num_motors
        
        # Get first arm properties (assuming all similar)
        arm = self.config.arms[0]
        
        # Bending stress (simplified)
        bending_stress = (force_per_arm * arm.length_m) / (arm.cross_section_area_m2 * 1e6)  # Convert to MPa
        
        # Stress percentage relative to material limit
        stress_percentage = (bending_stress / arm.max_bending_stress_mpa) * 100
        
        return min(100, max(0, stress_percentage))
    
    def _calculate_battery_stress(self, temperature: float) -> float:
        """Calculate battery stress"""
        
        # Battery optimal temperature: 20-30°C
        if 20 <= temperature <= 30:
            return 0.0
        
        if temperature < 20:
            stress = (20 - temperature) * 2
        else:
            stress = (temperature - 30) * 3
        
        return min(100, max(0, stress))
    
    def calculate_component_health(self, 
                                   flight_time_hours: float,
                                   avg_stress: float) -> Dict:
        """Calculate component health based on usage"""
        
        results = {}
        
        # Motor health
        if self.config.motors:
            motor = self.config.motors[0]
            motor_usage = flight_time_hours / motor.expected_lifetime_hours
            motor_wear = motor_usage * (1 + avg_stress / 100)  # Stress accelerates wear
            motor_health = max(0, 100 - (motor_wear * 100))
            
            results['motor_health'] = motor_health
            results['motor_remaining_hours'] = max(0, motor.expected_lifetime_hours * (1 - motor_wear))
        
        # Battery health (cycle-based)
        if self.config.battery:
            # Estimate cycles from flight time
            estimated_cycles = flight_time_hours / 0.5  # Assume 30min per cycle
            cycle_usage = estimated_cycles / self.config.battery.expected_cycle_life
            battery_wear = cycle_usage * (1 + avg_stress / 200)
            battery_health = max(0, 100 - (battery_wear * 100))
            
            results['battery_health'] = battery_health
            results['battery_remaining_cycles'] = max(0, self.config.battery.expected_cycle_life * (1 - battery_wear))
        
        # Frame/arm health (fatigue-based)
        if self.config.arms:
            # Frame degrades with stress cycles
            stress_cycles = flight_time_hours * 60  # Assume 1 cycle per minute
            fatigue_factor = (avg_stress / 100) ** 2  # Quadratic relationship
            frame_wear = (stress_cycles / 100000) * fatigue_factor  # 100k cycles nominal
            frame_health = max(0, 100 - (frame_wear * 100))
            
            results['frame_health'] = frame_health
        
        results['overall_health'] = np.mean([v for k, v in results.items() if 'health' in k])
        
        return results