"""
Adaptive Path Planning Module for Dynamic Trajectory Optimization
Implements path planning algorithms that adapt to detected anomalies
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import splprep, splev


class AdaptivePathPlanner:
    """Dynamic path planner that optimizes trajectories based on drone health"""
    
    def __init__(self, safety_margin: float = 10.0):
        """
        Initialize the adaptive path planner
        
        Args:
            safety_margin: Safety margin in meters for obstacle avoidance
        """
        self.safety_margin = safety_margin
        self.current_path = None
        self.waypoints = []
        
    def plan_initial_path(self, start: Tuple[float, float, float], 
                          end: Tuple[float, float, float],
                          obstacles: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Plan initial optimal path from start to end
        
        Args:
            start: Starting position (x, y, z)
            end: End position (x, y, z)
            obstacles: List of obstacles (not implemented in basic version)
            
        Returns:
            Array of waypoints defining the path
        """
        # Simple straight-line path for basic implementation
        num_points = 20
        
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        z = np.linspace(start[2], end[2], num_points)
        
        self.current_path = np.column_stack([x, y, z])
        self.waypoints = self.current_path.copy()
        
        return self.current_path
    
    def optimize_trajectory(self, current_position: Tuple[float, float, float],
                           risk_assessment: Dict,
                           destination: Tuple[float, float, float]) -> np.ndarray:
        """
        Dynamically optimize trajectory based on current risk assessment
        
        Args:
            current_position: Current drone position (x, y, z)
            risk_assessment: Risk assessment from anomaly detector
            destination: Final destination (x, y, z)
            
        Returns:
            Optimized path as array of waypoints
        """
        risk_level = risk_assessment.get('risk_level', 'LOW')
        
        if risk_level == "CRITICAL":
            # Emergency landing path - straight down with slight forward movement
            return self._emergency_landing_path(current_position)
        
        elif risk_level == "HIGH":
            # Conservative path - reduce altitude and head to nearest safe zone
            return self._conservative_path(current_position, destination)
        
        elif risk_level == "MEDIUM":
            # Cautious path - reduce speed, increase waypoint density
            return self._cautious_path(current_position, destination)
        
        else:
            # Normal optimal path
            return self._optimal_path(current_position, destination)
    
    def _emergency_landing_path(self, current_position: Tuple[float, float, float]) -> np.ndarray:
        """Generate emergency landing path"""
        x, y, z = current_position
        
        # Create path that descends quickly
        num_points = 10
        landing_x = x + 5  # Slight forward movement
        landing_y = y + 5
        
        x_path = np.linspace(x, landing_x, num_points)
        y_path = np.linspace(y, landing_y, num_points)
        z_path = np.linspace(z, 0, num_points)  # Descend to ground
        
        return np.column_stack([x_path, y_path, z_path])
    
    def _conservative_path(self, current_position: Tuple[float, float, float],
                          destination: Tuple[float, float, float]) -> np.ndarray:
        """Generate conservative path with reduced altitude"""
        x_curr, y_curr, z_curr = current_position
        x_dest, y_dest, z_dest = destination
        
        # Lower altitude for safer flight
        safe_altitude = min(z_curr * 0.7, z_dest)
        
        num_points = 25
        
        # Create multi-segment path: descend, travel, ascend if needed
        x_path = np.linspace(x_curr, x_dest, num_points)
        y_path = np.linspace(y_curr, y_dest, num_points)
        
        # Create altitude profile
        z_path = np.ones(num_points) * safe_altitude
        z_path[0] = z_curr
        z_path[-1] = z_dest
        
        # Smooth altitude transitions
        z_path[:5] = np.linspace(z_curr, safe_altitude, 5)
        z_path[-5:] = np.linspace(safe_altitude, z_dest, 5)
        
        return np.column_stack([x_path, y_path, z_path])
    
    def _cautious_path(self, current_position: Tuple[float, float, float],
                      destination: Tuple[float, float, float]) -> np.ndarray:
        """Generate cautious path with more waypoints"""
        num_points = 30
        
        x = np.linspace(current_position[0], destination[0], num_points)
        y = np.linspace(current_position[1], destination[1], num_points)
        z = np.linspace(current_position[2], destination[2], num_points)
        
        return np.column_stack([x, y, z])
    
    def _optimal_path(self, current_position: Tuple[float, float, float],
                     destination: Tuple[float, float, float]) -> np.ndarray:
        """Generate optimal direct path"""
        num_points = 20
        
        x = np.linspace(current_position[0], destination[0], num_points)
        y = np.linspace(current_position[1], destination[1], num_points)
        z = np.linspace(current_position[2], destination[2], num_points)
        
        return np.column_stack([x, y, z])
    
    def calculate_path_metrics(self, path: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for a given path
        
        Args:
            path: Array of waypoints
            
        Returns:
            Dictionary of path metrics
        """
        # Calculate total distance
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Calculate altitude changes
        altitude_changes = np.abs(np.diff(path[:, 2]))
        total_altitude_change = np.sum(altitude_changes)
        
        # Calculate smoothness (variance of direction changes)
        if len(path) > 2:
            vectors = np.diff(path, axis=0)
            angles = []
            for i in range(len(vectors) - 1):
                cos_angle = np.dot(vectors[i], vectors[i+1]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i+1]) + 1e-10
                )
                angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
            smoothness = 1.0 / (1.0 + np.var(angles))
        else:
            smoothness = 1.0
        
        return {
            'total_distance': float(total_distance),
            'total_altitude_change': float(total_altitude_change),
            'smoothness': float(smoothness),
            'num_waypoints': len(path)
        }
