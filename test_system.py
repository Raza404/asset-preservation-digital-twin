"""
Test suite for the Digital Twin System
"""

import numpy as np
import unittest
from anomaly_detection import AnomalyDetector
from path_planning import AdaptivePathPlanner
from digital_twin_engine import DigitalTwinEngine
from telemetry_simulator import TelemetrySimulator


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = AnomalyDetector(contamination=0.1)
        self.simulator = TelemetrySimulator(seed=42)
        
    def test_initialization(self):
        """Test detector initialization"""
        self.assertFalse(self.detector.is_trained)
        self.assertEqual(len(self.detector.feature_names), 6)
    
    def test_training(self):
        """Test model training"""
        training_data = self.simulator.generate_training_dataset(100)
        self.detector.train(training_data)
        self.assertTrue(self.detector.is_trained)
    
    def test_prediction_normal(self):
        """Test prediction on normal data"""
        training_data = self.simulator.generate_training_dataset(100)
        self.detector.train(training_data)
        
        normal_sample = self.simulator.generate_normal_telemetry(1)[0]
        prediction, score = self.detector.predict(normal_sample)
        
        # Normal samples should have lower anomaly scores
        self.assertLess(score, 0.8)
    
    def test_prediction_anomalous(self):
        """Test prediction on anomalous data"""
        training_data = self.simulator.generate_training_dataset(100)
        self.detector.train(training_data)
        
        anomalous_sample = self.simulator.generate_anomalous_telemetry(1, 'battery')[0]
        prediction, score = self.detector.predict(anomalous_sample)
        
        # Anomalous samples should be detected
        self.assertIsNotNone(score)
    
    def test_failure_risk_detection(self):
        """Test failure risk assessment"""
        training_data = self.simulator.generate_training_dataset(100)
        self.detector.train(training_data)
        
        sample = self.simulator.generate_normal_telemetry(1)[0]
        risk = self.detector.detect_failure_risk(sample)
        
        self.assertIn('is_anomaly', risk)
        self.assertIn('anomaly_score', risk)
        self.assertIn('risk_level', risk)
        self.assertIn('recommendation', risk)


class TestPathPlanning(unittest.TestCase):
    """Test cases for path planning module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.planner = AdaptivePathPlanner(safety_margin=10.0)
    
    def test_initial_path_planning(self):
        """Test initial path generation"""
        start = (0.0, 0.0, 50.0)
        end = (100.0, 100.0, 50.0)
        
        path = self.planner.plan_initial_path(start, end)
        
        self.assertIsNotNone(path)
        self.assertEqual(path.shape[1], 3)  # 3D coordinates
        self.assertTrue(np.allclose(path[0], start))
        self.assertTrue(np.allclose(path[-1], end))
    
    def test_emergency_landing_path(self):
        """Test emergency landing path generation"""
        current_pos = (50.0, 50.0, 50.0)
        risk = {'risk_level': 'CRITICAL'}
        destination = (100.0, 100.0, 50.0)
        
        path = self.planner.optimize_trajectory(current_pos, risk, destination)
        
        # Emergency path should descend to ground
        self.assertEqual(path[-1][2], 0.0)
    
    def test_path_metrics(self):
        """Test path metrics calculation"""
        start = (0.0, 0.0, 0.0)
        end = (100.0, 0.0, 0.0)
        
        path = self.planner.plan_initial_path(start, end)
        metrics = self.planner.calculate_path_metrics(path)
        
        self.assertIn('total_distance', metrics)
        self.assertIn('smoothness', metrics)
        self.assertIn('num_waypoints', metrics)
        self.assertGreater(metrics['total_distance'], 0)


class TestDigitalTwinEngine(unittest.TestCase):
    """Test cases for digital twin engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = DigitalTwinEngine()
        self.simulator = TelemetrySimulator(seed=42)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertFalse(self.engine.is_initialized)
        
        training_data = self.simulator.generate_training_dataset(100)
        success = self.engine.initialize(training_data)
        
        self.assertTrue(success)
        self.assertTrue(self.engine.is_initialized)
    
    def test_mission_start(self):
        """Test mission start"""
        training_data = self.simulator.generate_training_dataset(100)
        self.engine.initialize(training_data)
        
        start = (0.0, 0.0, 50.0)
        end = (100.0, 100.0, 50.0)
        
        mission = self.engine.start_mission(start, end)
        
        self.assertEqual(mission['status'], 'ACTIVE')
        self.assertTrue(self.engine.mission_active)
    
    def test_telemetry_update(self):
        """Test telemetry processing"""
        training_data = self.simulator.generate_training_dataset(100)
        self.engine.initialize(training_data)
        
        telemetry = self.simulator.generate_normal_telemetry(1)[0]
        position = (10.0, 10.0, 50.0)
        
        update = self.engine.update_telemetry(telemetry, position)
        
        self.assertIn('risk_assessment', update)
        self.assertIn('timestamp', update)
    
    def test_trajectory_replanning(self):
        """Test trajectory replanning"""
        training_data = self.simulator.generate_training_dataset(100)
        self.engine.initialize(training_data)
        
        # Start mission
        start = (0.0, 0.0, 50.0)
        end = (100.0, 100.0, 50.0)
        self.engine.start_mission(start, end)
        
        # Update with telemetry
        telemetry = self.simulator.generate_normal_telemetry(1)[0]
        position = (10.0, 10.0, 50.0)
        self.engine.update_telemetry(telemetry, position)
        
        # Replan trajectory
        result = self.engine.replan_trajectory(position, end)
        
        self.assertIn('new_path', result)
        self.assertIn('path_metrics', result)


class TestTelemetrySimulator(unittest.TestCase):
    """Test cases for telemetry simulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = TelemetrySimulator(seed=42)
    
    def test_normal_telemetry_generation(self):
        """Test normal telemetry generation"""
        data = self.simulator.generate_normal_telemetry(10)
        
        self.assertEqual(data.shape, (10, 6))
        
        # Check values are in normal ranges
        self.assertTrue(np.all(data[:, 0] >= 80))  # Battery
        self.assertTrue(np.all(data[:, 0] <= 100))
    
    def test_anomalous_telemetry_generation(self):
        """Test anomalous telemetry generation"""
        data = self.simulator.generate_anomalous_telemetry(10, 'battery')
        
        self.assertEqual(data.shape, (10, 6))
        
        # Battery should be low
        self.assertTrue(np.any(data[:, 0] < 50))
    
    def test_training_dataset_generation(self):
        """Test training dataset generation"""
        data = self.simulator.generate_training_dataset(1000, contamination=0.1)
        
        self.assertEqual(len(data), 1000)
        self.assertEqual(data.shape[1], 6)
    
    def test_flight_sequence_simulation(self):
        """Test flight sequence simulation"""
        telemetry, positions = self.simulator.simulate_flight_sequence(duration=20)
        
        self.assertEqual(len(telemetry), 20)
        self.assertEqual(len(positions), 20)
        self.assertEqual(telemetry.shape[1], 6)


if __name__ == '__main__':
    unittest.main()
