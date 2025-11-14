"""
Trajectory Optimizer
Dynamically recalculates flight paths based on predicted failures, battery state, and component health
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization priorities"""
    SAFETY_FIRST = "safety_first"  # Minimize failure risk
    ENERGY_EFFICIENT = "energy_efficient"  # Maximize flight time
    MISSION_CRITICAL = "mission_critical"  # Complete mission at all costs
    BALANCED = "balanced"  # Balance all factors


class ActionType(Enum):
    """Possible drone actions"""
    REDUCE_SPEED = "reduce_speed"
    INCREASE_ALTITUDE = "increase_altitude"
    DECREASE_ALTITUDE = "decrease_altitude"
    HOVER = "hover"
    RETURN_TO_HOME = "return_to_home"
    EMERGENCY_LAND = "emergency_land"
    CHANGE_HEADING = "change_heading"
    REDUCE_THROTTLE = "reduce_throttle"
    AVOID_OBSTACLE = "avoid_obstacle"


@dataclass
class DroneState:
    """Current state of the drone"""
    latitude: float
    longitude: float
    altitude: float
    heading: float  # degrees
    speed: float  # m/s
    battery_voltage: float
    battery_remaining: float  # percentage
    component_health: Dict[str, float]  # component name -> health score (0-1)
    failure_probability: float  # 0-1
    timestamp: float


@dataclass
class Waypoint:
    """Target waypoint"""
    latitude: float
    longitude: float
    altitude: float
    tolerance: float = 5.0  # meters


@dataclass
class OptimizationResult:
    """Result of trajectory optimization"""
    action: ActionType
    new_speed: Optional[float] = None
    new_altitude: Optional[float] = None
    new_heading: Optional[float] = None
    throttle_reduction: Optional[float] = None  # percentage
    reason: str = ""
    urgency: float = 0.0  # 0-1, how urgent the action is
    estimated_benefit: Dict[str, float] = None  # what we expect to gain


class TrajectoryOptimizer:
    """
    Real-time trajectory optimization engine
    Analyzes current state and makes decisions to preserve the drone
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        failure_threshold: float = 0.25,
        battery_warning_threshold: float = 0.30,
        battery_critical_threshold: float = 0.20
    ):
        self.strategy = strategy
        self.failure_threshold = failure_threshold
        self.battery_warning = battery_warning_threshold
        self.battery_critical = battery_critical_threshold
        
        # Strategy weights
        self.weights = self._get_strategy_weights()
        
    def _get_strategy_weights(self) -> Dict[str, float]:
        """Get optimization weights based on strategy"""
        if self.strategy == OptimizationStrategy.SAFETY_FIRST:
            return {
                'safety': 0.7,
                'energy': 0.2,
                'mission': 0.1
            }
        elif self.strategy == OptimizationStrategy.ENERGY_EFFICIENT:
            return {
                'safety': 0.3,
                'energy': 0.6,
                'mission': 0.1
            }
        elif self.strategy == OptimizationStrategy.MISSION_CRITICAL:
            return {
                'safety': 0.2,
                'energy': 0.1,
                'mission': 0.7
            }
        else:  # BALANCED
            return {
                'safety': 0.4,
                'energy': 0.3,
                'mission': 0.3
            }
    
    def optimize(
        self,
        current_state: DroneState,
        target_waypoint: Optional[Waypoint] = None,
        wind_speed: float = 0.0,
        terrain_clearance: float = 50.0
    ) -> List[OptimizationResult]:
        """
        Main optimization function
        Returns a list of recommended actions in priority order
        """
        actions = []
        
        # 1. Check for critical failures (highest priority)
        failure_actions = self._check_failure_risk(current_state)
        if failure_actions:
            actions.extend(failure_actions)
            
        # 2. Check battery state
        battery_actions = self._optimize_battery_usage(current_state, target_waypoint)
        if battery_actions:
            actions.extend(battery_actions)
            
        # 3. Check component stress
        component_actions = self._reduce_component_stress(current_state)
        if component_actions:
            actions.extend(component_actions)
            
        # 4. Optimize for efficiency
        efficiency_actions = self._optimize_efficiency(current_state, wind_speed)
        if efficiency_actions:
            actions.extend(efficiency_actions)
        
        # Sort by urgency
        actions.sort(key=lambda x: x.urgency, reverse=True)
        
        return actions
    
    def _check_failure_risk(self, state: DroneState) -> List[OptimizationResult]:
        """Check for predicted failures and recommend preventive actions"""
        actions = []
        
        if state.failure_probability > 0.75:
            # Critical - immediate emergency landing
            actions.append(OptimizationResult(
                action=ActionType.EMERGENCY_LAND,
                reason=f"Critical failure probability: {state.failure_probability:.1%}",
                urgency=1.0,
                estimated_benefit={'survival_chance': 0.95}
            ))
            
        elif state.failure_probability > 0.50:
            # High risk - return to home immediately
            actions.append(OptimizationResult(
                action=ActionType.RETURN_TO_HOME,
                reason=f"High failure probability: {state.failure_probability:.1%}",
                urgency=0.9,
                estimated_benefit={'survival_chance': 0.85}
            ))
            
        elif state.failure_probability > self.failure_threshold:
            # Moderate risk - reduce stress on components
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_SPEED,
                new_speed=state.speed * 0.7,
                reason=f"Elevated failure risk: {state.failure_probability:.1%}. Reducing speed to decrease component stress.",
                urgency=0.7,
                estimated_benefit={
                    'failure_risk_reduction': 0.15,
                    'component_life_extension': 120  # seconds
                }
            ))
            
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_THROTTLE,
                throttle_reduction=20.0,
                reason="Reducing throttle to minimize vibration and motor stress",
                urgency=0.6,
                estimated_benefit={'motor_stress_reduction': 0.25}
            ))
        
        # Check individual component health
        for component, health in state.component_health.items():
            if health < 0.3:
                actions.append(OptimizationResult(
                    action=ActionType.HOVER,
                    reason=f"{component} health critically low ({health:.1%}). Hovering to assess situation.",
                    urgency=0.85,
                    estimated_benefit={'assessment_time': 10}  # seconds
                ))
                break
                
        return actions
    
    def _optimize_battery_usage(
        self,
        state: DroneState,
        target: Optional[Waypoint]
    ) -> List[OptimizationResult]:
        """Optimize battery usage and range"""
        actions = []
        
        if state.battery_remaining < self.battery_critical:
            # Critical battery - emergency land now
            actions.append(OptimizationResult(
                action=ActionType.EMERGENCY_LAND,
                reason=f"Critical battery level: {state.battery_remaining:.1%}",
                urgency=1.0,
                estimated_benefit={'safe_landing_chance': 0.90}
            ))
            
        elif state.battery_remaining < self.battery_warning:
            # Low battery - return to home
            if target:
                distance_to_target = self._calculate_distance(
                    state.latitude, state.longitude,
                    target.latitude, target.longitude
                )
                estimated_range = self._estimate_range(state)
                
                if distance_to_target * 1.5 > estimated_range:  # 1.5x safety margin
                    actions.append(OptimizationResult(
                        action=ActionType.RETURN_TO_HOME,
                        reason=f"Insufficient battery for mission completion. Estimated range: {estimated_range:.0f}m, Distance: {distance_to_target:.0f}m",
                        urgency=0.8,
                        estimated_benefit={'safe_return_chance': 0.85}
                    ))
            
            # Reduce speed for energy efficiency
            optimal_speed = self._calculate_optimal_speed_for_range(state)
            if optimal_speed < state.speed:
                actions.append(OptimizationResult(
                    action=ActionType.REDUCE_SPEED,
                    new_speed=optimal_speed,
                    reason=f"Reducing speed to {optimal_speed:.1f}m/s for optimal range extension",
                    urgency=0.6,
                    estimated_benefit={
                        'range_extension': 15,  # percentage
                        'flight_time_extension': 120  # seconds
                    }
                ))
        
        # Continuous efficiency optimization
        if state.battery_remaining < 0.60 and state.speed > 8.0:
            # Moderate battery, reduce to cruise speed
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_SPEED,
                new_speed=7.0,
                reason="Reducing to cruise speed for better energy efficiency",
                urgency=0.3,
                estimated_benefit={'energy_savings': 0.20}
            ))
            
        return actions
    
    def _reduce_component_stress(self, state: DroneState) -> List[OptimizationResult]:
        """Reduce mechanical stress on components"""
        actions = []
        
        # Check motor/ESC stress
        motor_health = state.component_health.get('motor', 1.0)
        esc_health = state.component_health.get('esc', 1.0)
        
        avg_motor_health = (motor_health + esc_health) / 2
        
        if avg_motor_health < 0.6:
            # Reduce throttle to decrease motor stress
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_THROTTLE,
                throttle_reduction=25.0,
                reason=f"Motor/ESC health degraded ({avg_motor_health:.1%}). Reducing load.",
                urgency=0.65,
                estimated_benefit={
                    'component_life_extension': 180,  # seconds
                    'heat_reduction': 0.30
                }
            ))
            
            # Reduce speed to minimize vibration
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_SPEED,
                new_speed=state.speed * 0.65,
                reason="Reducing speed to minimize vibration and component wear",
                urgency=0.6,
                estimated_benefit={'vibration_reduction': 0.40}
            ))
        
        # Check battery health (voltage sag)
        if state.battery_voltage < 3.3 * 4:  # 4S battery minimum
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_THROTTLE,
                throttle_reduction=30.0,
                reason=f"Battery voltage low ({state.battery_voltage:.2f}V). Reducing load to prevent damage.",
                urgency=0.75,
                estimated_benefit={'battery_preservation': 0.25}
            ))
            
        return actions
    
    def _optimize_efficiency(
        self,
        state: DroneState,
        wind_speed: float
    ) -> List[OptimizationResult]:
        """Optimize for energy efficiency"""
        actions = []
        
        # Adjust altitude based on wind
        if wind_speed > 8.0 and state.altitude < 100:
            # High wind at low altitude - climb for better conditions
            actions.append(OptimizationResult(
                action=ActionType.INCREASE_ALTITUDE,
                new_altitude=min(state.altitude + 30, 120),
                reason=f"High wind speed ({wind_speed:.1f}m/s) at low altitude. Climbing for efficiency.",
                urgency=0.4,
                estimated_benefit={'energy_savings': 0.15}
            ))
        
        # Speed optimization
        if state.speed > 12.0:
            # Very high speed - inefficient
            actions.append(OptimizationResult(
                action=ActionType.REDUCE_SPEED,
                new_speed=10.0,
                reason="Speed above optimal range. Reducing for efficiency.",
                urgency=0.3,
                estimated_benefit={'energy_savings': 0.25}
            ))
            
        return actions
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two coordinates (simplified)"""
        # Haversine formula (simplified for small distances)
        R = 6371000  # Earth radius in meters
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _estimate_range(self, state: DroneState) -> float:
        """Estimate remaining range in meters"""
        # Simplified model: range proportional to battery remaining
        # Assume 10km max range at full battery, 5m/s average speed
        max_range = 10000  # meters
        return max_range * state.battery_remaining
    
    def _calculate_optimal_speed_for_range(self, state: DroneState) -> float:
        """Calculate optimal speed for maximum range"""
        # Optimal cruise speed is typically 60-70% of max speed
        # For most drones this is around 7-8 m/s
        return 7.0
    
    def should_abort_mission(self, state: DroneState) -> Tuple[bool, str]:
        """
        Determine if mission should be aborted
        Returns (should_abort, reason)
        """
        if state.failure_probability > 0.65:
            return True, f"Failure probability too high: {state.failure_probability:.1%}"
        
        if state.battery_remaining < self.battery_critical:
            return True, f"Critical battery level: {state.battery_remaining:.1%}"
        
        critical_components = [
            name for name, health in state.component_health.items()
            if health < 0.25
        ]
        if critical_components:
            return True, f"Critical component failure: {', '.join(critical_components)}"
        
        return False, ""
    
    def generate_safe_trajectory(
        self,
        current_state: DroneState,
        target: Waypoint,
        intermediate_points: int = 5
    ) -> List[Waypoint]:
        """
        Generate a safe trajectory with intermediate waypoints
        that optimize for current constraints
        """
        waypoints = []
        
        # Calculate optimal altitude based on conditions
        optimal_altitude = self._calculate_optimal_altitude(current_state)
        
        # Generate intermediate waypoints
        for i in range(1, intermediate_points + 1):
            fraction = i / (intermediate_points + 1)
            
            wp = Waypoint(
                latitude=current_state.latitude + (target.latitude - current_state.latitude) * fraction,
                longitude=current_state.longitude + (target.longitude - current_state.longitude) * fraction,
                altitude=optimal_altitude
            )
            waypoints.append(wp)
        
        # Add final target
        waypoints.append(target)
        
        return waypoints
    
    def _calculate_optimal_altitude(self, state: DroneState) -> float:
        """Calculate optimal altitude based on conditions"""
        base_altitude = 50.0  # Safe minimum
        
        # Increase if battery is good and we want efficiency
        if state.battery_remaining > 0.5:
            base_altitude = 75.0
        
        # Decrease if battery is low or components stressed
        if state.battery_remaining < 0.3 or state.failure_probability > 0.3:
            base_altitude = 40.0
        
        return base_altitude
