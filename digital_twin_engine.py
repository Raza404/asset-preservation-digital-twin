"""
Digital Twin Engine - Core real-time monitoring and control system
Integrates anomaly detection and path planning for asset preservation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

from anomaly_detection import AnomalyDetector
from path_planning import AdaptivePathPlanner


class DigitalTwinEngine:
    """Main digital twin engine for real-time drone monitoring and control"""
    
    def __init__(self, contamination: float = 0.1, safety_margin: float = 10.0):
        """
        Initialize the digital twin engine
        
        Args:
            contamination: Expected anomaly rate for ML model
            safety_margin: Safety margin for path planning in meters
        """
        self.anomaly_detector = AnomalyDetector(contamination=contamination)
        self.path_planner = AdaptivePathPlanner(safety_margin=safety_margin)
        
        self.current_state = {
            'position': (0.0, 0.0, 0.0),
            'velocity': (0.0, 0.0, 0.0),
            'telemetry': None,
            'risk_assessment': None,
            'current_path': None
        }
        
        self.history = {
            'telemetry': [],
            'risk_assessments': [],
            'positions': [],
            'timestamps': []
        }
        
        self.is_initialized = False
        self.mission_active = False
        
    def initialize(self, training_data: np.ndarray) -> bool:
        """
        Initialize the system with training data
        
        Args:
            training_data: Historical telemetry data for training anomaly detector
            
        Returns:
            True if initialization successful
        """
        try:
            self.anomaly_detector.train(training_data)
            self.is_initialized = True
            print("✓ Digital Twin Engine initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            return False
    
    def start_mission(self, start_position: Tuple[float, float, float],
                     end_position: Tuple[float, float, float]) -> Dict:
        """
        Start a new mission with initial path planning
        
        Args:
            start_position: Starting position (x, y, z)
            end_position: Destination position (x, y, z)
            
        Returns:
            Mission info dictionary
        """
        if not self.is_initialized:
            raise RuntimeError("Engine must be initialized before starting mission")
        
        # Plan initial path
        initial_path = self.path_planner.plan_initial_path(start_position, end_position)
        
        self.current_state['position'] = start_position
        self.current_state['current_path'] = initial_path
        self.mission_active = True
        
        path_metrics = self.path_planner.calculate_path_metrics(initial_path)
        
        print(f"✓ Mission started")
        print(f"  From: {start_position}")
        print(f"  To: {end_position}")
        print(f"  Path distance: {path_metrics['total_distance']:.2f}m")
        
        return {
            'status': 'ACTIVE',
            'start_position': start_position,
            'end_position': end_position,
            'initial_path': initial_path,
            'path_metrics': path_metrics
        }
    
    def update_telemetry(self, telemetry: np.ndarray, 
                        current_position: Tuple[float, float, float]) -> Dict:
        """
        Process new telemetry data and update system state
        
        Args:
            telemetry: Current telemetry reading
            current_position: Current drone position
            
        Returns:
            System update dictionary with risk assessment and path updates
        """
        if not self.is_initialized:
            raise RuntimeError("Engine must be initialized")
        
        timestamp = datetime.now()
        
        # Detect anomalies
        risk_assessment = self.anomaly_detector.detect_failure_risk(telemetry)
        
        # Update state
        self.current_state['telemetry'] = telemetry
        self.current_state['position'] = current_position
        self.current_state['risk_assessment'] = risk_assessment
        
        # Store history
        self.history['telemetry'].append(telemetry)
        self.history['risk_assessments'].append(risk_assessment)
        self.history['positions'].append(current_position)
        self.history['timestamps'].append(timestamp)
        
        # Prepare response
        response = {
            'timestamp': timestamp.isoformat(),
            'position': current_position,
            'risk_assessment': risk_assessment,
            'path_update_required': False,
            'new_path': None
        }
        
        # Check if path update is needed
        if risk_assessment['risk_level'] in ['MEDIUM', 'HIGH', 'CRITICAL']:
            response['path_update_required'] = True
            print(f"⚠ Risk level: {risk_assessment['risk_level']}")
            print(f"  Anomaly score: {risk_assessment['anomaly_score']:.3f}")
            print(f"  Recommendation: {risk_assessment['recommendation']}")
        
        return response
    
    def replan_trajectory(self, current_position: Tuple[float, float, float],
                         destination: Tuple[float, float, float]) -> Dict:
        """
        Dynamically replan trajectory based on current risk
        
        Args:
            current_position: Current drone position
            destination: Target destination
            
        Returns:
            New trajectory plan
        """
        if not self.current_state['risk_assessment']:
            raise RuntimeError("No risk assessment available")
        
        # Get optimized path
        new_path = self.path_planner.optimize_trajectory(
            current_position,
            self.current_state['risk_assessment'],
            destination
        )
        
        self.current_state['current_path'] = new_path
        
        path_metrics = self.path_planner.calculate_path_metrics(new_path)
        
        risk_level = self.current_state['risk_assessment']['risk_level']
        print(f"✓ Trajectory replanned for {risk_level} risk")
        print(f"  New path distance: {path_metrics['total_distance']:.2f}m")
        print(f"  Waypoints: {path_metrics['num_waypoints']}")
        
        return {
            'new_path': new_path,
            'path_metrics': path_metrics,
            'risk_level': risk_level
        }
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary with complete system state
        """
        status = {
            'initialized': self.is_initialized,
            'mission_active': self.mission_active,
            'current_position': self.current_state['position'],
            'current_risk': self.current_state['risk_assessment'],
            'total_updates': len(self.history['timestamps'])
        }
        
        if len(self.history['risk_assessments']) > 0:
            risk_levels = [r['risk_level'] for r in self.history['risk_assessments']]
            anomaly_scores = [r['anomaly_score'] for r in self.history['risk_assessments']]
            
            status['risk_summary'] = {
                'low_count': risk_levels.count('LOW'),
                'medium_count': risk_levels.count('MEDIUM'),
                'high_count': risk_levels.count('HIGH'),
                'critical_count': risk_levels.count('CRITICAL'),
                'avg_anomaly_score': float(np.mean(anomaly_scores)),
                'max_anomaly_score': float(np.max(anomaly_scores))
            }
        
        return status
    
    def stop_mission(self) -> Dict:
        """
        Stop the current mission
        
        Returns:
            Mission summary
        """
        self.mission_active = False
        
        summary = {
            'total_telemetry_updates': len(self.history['timestamps']),
            'final_position': self.current_state['position'],
            'final_risk': self.current_state['risk_assessment']
        }
        
        if len(self.history['risk_assessments']) > 0:
            summary['risk_summary'] = self.get_system_status()['risk_summary']
        
        print("✓ Mission stopped")
        return summary
