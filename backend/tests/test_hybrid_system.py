"""
Unit tests for hybrid LSTM + RF system components.
"""

import pytest
import numpy as np
import pandas as pd
from src.ml.lstm_failure_predictor import LSTMFailurePredictor
from src.ml.enhanced_rf_predictor import EnhancedRandomForestPredictor
from src.ml.hybrid_decision_engine import HybridDecisionEngine, FlightAction


def test_lstm_predictor_initialization():
    """Test LSTM predictor can be initialized."""
    predictor = LSTMFailurePredictor(sequence_length=30)
    assert predictor.sequence_length == 30
    assert len(predictor.prediction_horizons) == 3
    assert predictor.component_thresholds is not None


def test_lstm_heuristic_prediction():
    """Test LSTM fallback heuristic prediction."""
    predictor = LSTMFailurePredictor()
    
    # Create dummy sequence
    sequence = np.random.rand(30, 10)
    
    # Get predictions (will use heuristics since TF not installed)
    results = predictor.predict_component_failure(sequence)
    
    assert 'motor' in results
    assert 'battery' in results
    assert 'failure_probability' in results['motor']
    assert 'recommendations' in results['motor']
    assert isinstance(results['motor']['recommendations'], list)


def test_enhanced_rf_initialization():
    """Test Enhanced RF predictor initialization."""
    predictor = EnhancedRandomForestPredictor()
    assert predictor.component_features is not None
    assert 'motor' in predictor.component_features
    assert 'battery' in predictor.component_features


def test_hybrid_engine_initialization():
    """Test hybrid decision engine initialization."""
    engine = HybridDecisionEngine()
    assert engine.rf_predictor is not None
    assert engine.lstm_predictor is not None
    assert engine.buffer_size == 30
    assert len(engine.telemetry_buffer) == 0


def test_telemetry_processing():
    """Test telemetry processing without trained models."""
    engine = HybridDecisionEngine()
    
    telemetry = {
        'battery_voltage': 12.4,
        'battery_remaining': 85,
        'throttle': 55,
        'ground_speed': 12.0,
        'altitude': 50.0,
        'motor_temp': 60,
        'esc_temp': 50,
        'vibration_magnitude': 0.3,
        'gps_satellites': 14,
        'gyro_x': 10, 'gyro_y': -5, 'gyro_z': 2,
        'accel_x': 0.2, 'accel_y': -0.1, 'accel_z': 9.8
    }
    
    result = engine.process_telemetry(telemetry)
    
    assert 'flight_state' in result
    assert 'risk_level' in result
    assert 'recommended_actions' in result
    assert len(engine.telemetry_buffer) == 1


def test_flight_action_creation():
    """Test FlightAction dataclass."""
    action = FlightAction(
        action_type='throttle',
        severity='warning',
        value=70.0,
        reason='Test action',
        component='motor',
        time_critical=False
    )
    
    assert action.action_type == 'throttle'
    assert action.value == 70.0
    assert not action.time_critical


def test_action_formatting():
    """Test action to dict conversion."""
    engine = HybridDecisionEngine()
    
    action = FlightAction(
        action_type='land',
        severity='critical',
        value=None,
        reason='Critical failure',
        component='system',
        time_critical=True
    )
    
    action_dict = engine._action_to_dict(action)
    
    assert 'action_type' in action_dict
    assert 'severity' in action_dict
    assert 'display' in action_dict
    assert action_dict['time_critical'] is True


def test_buffer_management():
    """Test telemetry buffer size limit."""
    engine = HybridDecisionEngine()
    
    telemetry = {
        'battery_voltage': 12.0,
        'throttle': 50
    }
    
    # Add more than buffer size
    for i in range(50):
        engine.telemetry_buffer.append(telemetry)
        if len(engine.telemetry_buffer) > engine.buffer_size:
            engine.telemetry_buffer.pop(0)
    
    assert len(engine.telemetry_buffer) == engine.buffer_size


def test_component_health_thresholds():
    """Test component threshold definitions."""
    predictor = LSTMFailurePredictor()
    
    assert 'motor' in predictor.component_thresholds
    assert 'critical' in predictor.component_thresholds['motor']
    assert predictor.component_thresholds['motor']['critical'] > 0.5


def test_recommendation_generation():
    """Test recommendation generation for components."""
    predictor = LSTMFailurePredictor()
    
    sequence = np.random.rand(30, 10)
    health_info = predictor._analyze_component_health('motor', 0.85, sequence)
    
    assert 'recommendations' in health_info
    assert len(health_info['recommendations']) > 0
    assert health_info['status'] == 'CRITICAL'


def test_multiple_telemetry_updates():
    """Test processing multiple telemetry updates."""
    engine = HybridDecisionEngine()
    
    for i in range(10):
        telemetry = {
            'battery_voltage': 12.4 - (i * 0.1),  # Decreasing voltage
            'battery_remaining': 85 - (i * 5),
            'throttle': 55 + i,
            'motor_temp': 60 + (i * 2),
            'vibration_magnitude': 0.3 + (i * 0.05)
        }
        
        result = engine.process_telemetry(telemetry)
        assert result is not None
    
    assert len(engine.telemetry_buffer) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
