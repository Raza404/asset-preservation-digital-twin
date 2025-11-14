"""
Tests for Trajectory Optimization System
"""
import pytest
from src.processing.trajectory_optimizer import (
    TrajectoryOptimizer,
    DroneState,
    Waypoint,
    ActionType,
    OptimizationStrategy
)


def test_trajectory_optimizer_initialization():
    """Test that optimizer initializes correctly"""
    optimizer = TrajectoryOptimizer(strategy=OptimizationStrategy.SAFETY_FIRST)
    assert optimizer is not None
    assert optimizer.failure_threshold == 0.25


def test_high_failure_probability_triggers_emergency():
    """Test that high failure probability triggers emergency landing"""
    optimizer = TrajectoryOptimizer()
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=10.0,
        battery_voltage=14.8,
        battery_remaining=0.80,
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.85,  # Very high!
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    assert len(actions) > 0
    assert actions[0].action == ActionType.EMERGENCY_LAND
    assert actions[0].urgency == 1.0


def test_low_battery_triggers_rtm():
    """Test that low battery triggers return to home"""
    optimizer = TrajectoryOptimizer(battery_critical_threshold=0.20)
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=10.0,
        battery_voltage=12.8,
        battery_remaining=0.15,  # Critical!
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.10,
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    assert len(actions) > 0
    assert actions[0].action == ActionType.EMERGENCY_LAND


def test_moderate_risk_reduces_speed():
    """Test that moderate failure risk reduces speed"""
    optimizer = TrajectoryOptimizer(failure_threshold=0.25)
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=12.0,
        battery_voltage=15.0,
        battery_remaining=0.70,
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.35,  # Above threshold
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    assert len(actions) > 0
    # Should recommend speed reduction
    speed_actions = [a for a in actions if a.action == ActionType.REDUCE_SPEED]
    assert len(speed_actions) > 0
    assert speed_actions[0].new_speed < state.speed


def test_component_health_degradation():
    """Test that degraded component health triggers actions"""
    optimizer = TrajectoryOptimizer()
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=10.0,
        battery_voltage=15.0,
        battery_remaining=0.80,
        component_health={'motor': 0.25, 'esc': 0.30},  # Degraded!
        failure_probability=0.15,
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    assert len(actions) > 0
    # Should recommend hover or reduce throttle
    assert any(a.action == ActionType.HOVER for a in actions)


def test_optimal_trajectory_generation():
    """Test generation of safe trajectory"""
    optimizer = TrajectoryOptimizer()
    
    current = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=10.0,
        battery_voltage=15.0,
        battery_remaining=0.80,
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.10,
        timestamp=1234567890.0
    )
    
    target = Waypoint(latitude=40.7200, longitude=-74.0000, altitude=60.0)
    
    waypoints = optimizer.generate_safe_trajectory(current, target, intermediate_points=3)
    
    assert len(waypoints) == 4  # 3 intermediate + 1 target
    assert waypoints[-1].latitude == target.latitude
    assert waypoints[-1].longitude == target.longitude


def test_mission_abort_decision():
    """Test mission abort logic"""
    optimizer = TrajectoryOptimizer()
    
    # Critical failure probability
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=10.0,
        battery_voltage=15.0,
        battery_remaining=0.80,
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.70,
        timestamp=1234567890.0
    )
    
    should_abort, reason = optimizer.should_abort_mission(state)
    
    assert should_abort == True
    assert "failure probability" in reason.lower()


def test_energy_efficient_strategy():
    """Test energy-efficient optimization strategy"""
    optimizer = TrajectoryOptimizer(strategy=OptimizationStrategy.ENERGY_EFFICIENT)
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=15.0,  # High speed
        battery_voltage=14.5,
        battery_remaining=0.55,
        component_health={'motor': 1.0, 'esc': 1.0},
        failure_probability=0.10,
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    # Should recommend speed reduction for efficiency
    speed_actions = [a for a in actions if a.action == ActionType.REDUCE_SPEED]
    assert len(speed_actions) > 0


def test_normal_conditions_no_urgent_actions():
    """Test that normal conditions don't trigger urgent actions"""
    optimizer = TrajectoryOptimizer()
    
    state = DroneState(
        latitude=40.7128,
        longitude=-74.0060,
        altitude=50.0,
        heading=90.0,
        speed=8.0,
        battery_voltage=15.8,
        battery_remaining=0.85,
        component_health={'motor': 1.0, 'esc': 1.0, 'battery': 1.0},
        failure_probability=0.05,
        timestamp=1234567890.0
    )
    
    actions = optimizer.optimize(state)
    
    # May have some efficiency recommendations but nothing urgent
    if actions:
        assert all(a.urgency < 0.7 for a in actions)
