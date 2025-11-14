"""
Real-Time Decision Maker
Integrates failure prediction with trajectory optimization to make autonomous decisions
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from ..processing.trajectory_optimizer import (
    TrajectoryOptimizer,
    DroneState,
    Waypoint,
    OptimizationResult,
    ActionType,
    OptimizationStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionLog:
    """Log of a decision made by the system"""
    timestamp: float
    drone_state: Dict
    actions_recommended: List[Dict]
    action_taken: Optional[str]
    reason: str
    urgency: float
    outcome: Optional[str] = None


class RealTimeDecisionMaker:
    """
    Makes real-time decisions based on:
    1. ML failure predictions
    2. Current telemetry
    3. Mission objectives
    4. Component health
    """
    
    def __init__(
        self,
        optimizer: TrajectoryOptimizer,
        auto_execute: bool = False,
        log_decisions: bool = True
    ):
        self.optimizer = optimizer
        self.auto_execute = auto_execute  # If True, sends commands directly to drone
        self.log_decisions = log_decisions
        self.decision_history: List[DecisionLog] = []
        
        # Thresholds for action
        self.action_urgency_threshold = 0.5  # Only act on actions with urgency > 0.5
        
    def process_telemetry(
        self,
        telemetry: Dict,
        failure_prediction: float,
        component_health: Dict[str, float],
        target_waypoint: Optional[Waypoint] = None
    ) -> Tuple[List[OptimizationResult], Optional[str]]:
        """
        Process incoming telemetry and make decisions
        
        Returns:
            (recommended_actions, command_to_send)
        """
        # Build current drone state
        current_state = self._build_drone_state(
            telemetry,
            failure_prediction,
            component_health
        )
        
        # Get optimization recommendations
        actions = self.optimizer.optimize(
            current_state,
            target_waypoint,
            wind_speed=telemetry.get('wind_speed', 0.0)
        )
        
        # Check if we should abort mission
        should_abort, abort_reason = self.optimizer.should_abort_mission(current_state)
        
        if should_abort:
            logger.critical(f"MISSION ABORT RECOMMENDED: {abort_reason}")
            # Force emergency action
            actions.insert(0, OptimizationResult(
                action=ActionType.EMERGENCY_LAND,
                reason=abort_reason,
                urgency=1.0
            ))
        
        # Log the decision
        if self.log_decisions and actions:
            self._log_decision(current_state, actions)
        
        # Determine which action to take
        command = None
        if actions and actions[0].urgency >= self.action_urgency_threshold:
            command = self._generate_command(actions[0], current_state)
            
            if self.auto_execute and command:
                logger.info(f"AUTO-EXECUTING: {command}")
                # In real implementation, this would send to drone via MQTT/MAVLink
        
        return actions, command
    
    def _build_drone_state(
        self,
        telemetry: Dict,
        failure_prob: float,
        component_health: Dict[str, float]
    ) -> DroneState:
        """Build DroneState from telemetry"""
        return DroneState(
            latitude=telemetry.get('latitude', 0.0),
            longitude=telemetry.get('longitude', 0.0),
            altitude=telemetry.get('altitude', 0.0),
            heading=telemetry.get('heading', 0.0),
            speed=telemetry.get('ground_speed', 0.0),
            battery_voltage=telemetry.get('battery_voltage', 12.6),
            battery_remaining=telemetry.get('battery_remaining', 100.0) / 100.0,
            component_health=component_health,
            failure_probability=failure_prob,
            timestamp=telemetry.get('timestamp', datetime.now().timestamp())
        )
    
    def _generate_command(
        self,
        action: OptimizationResult,
        state: DroneState
    ) -> Optional[str]:
        """
        Generate drone command from optimization result
        Returns command string in format: "ACTION:PARAMS"
        """
        if action.action == ActionType.REDUCE_SPEED:
            return f"SET_SPEED:{action.new_speed:.2f}"
        
        elif action.action == ActionType.REDUCE_THROTTLE:
            return f"REDUCE_THROTTLE:{action.throttle_reduction:.0f}"
        
        elif action.action == ActionType.INCREASE_ALTITUDE:
            return f"SET_ALTITUDE:{action.new_altitude:.1f}"
        
        elif action.action == ActionType.DECREASE_ALTITUDE:
            return f"SET_ALTITUDE:{action.new_altitude:.1f}"
        
        elif action.action == ActionType.HOVER:
            return "HOVER"
        
        elif action.action == ActionType.RETURN_TO_HOME:
            return "RTH"
        
        elif action.action == ActionType.EMERGENCY_LAND:
            return "EMERGENCY_LAND"
        
        elif action.action == ActionType.CHANGE_HEADING:
            return f"SET_HEADING:{action.new_heading:.1f}"
        
        return None
    
    def _log_decision(
        self,
        state: DroneState,
        actions: List[OptimizationResult]
    ):
        """Log decision for analysis"""
        log = DecisionLog(
            timestamp=state.timestamp,
            drone_state=asdict(state),
            actions_recommended=[
                {
                    'action': act.action.value,
                    'reason': act.reason,
                    'urgency': act.urgency,
                    'benefit': act.estimated_benefit
                }
                for act in actions[:3]  # Log top 3 actions
            ],
            action_taken=actions[0].action.value if actions else None,
            reason=actions[0].reason if actions else "",
            urgency=actions[0].urgency if actions else 0.0
        )
        
        self.decision_history.append(log)
        
        # Keep only last 1000 decisions in memory
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_decision_summary(self) -> Dict:
        """Get summary of recent decisions"""
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'recent_actions': []
            }
        
        recent = self.decision_history[-10:]
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_actions': [
                {
                    'timestamp': log.timestamp,
                    'action': log.action_taken,
                    'reason': log.reason,
                    'urgency': log.urgency
                }
                for log in recent
            ],
            'action_distribution': self._get_action_distribution()
        }
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken"""
        distribution = {}
        for log in self.decision_history:
            action = log.action_taken
            if action:
                distribution[action] = distribution.get(action, 0) + 1
        return distribution
    
    def export_decision_log(self, filepath: str):
        """Export decision history to JSON"""
        with open(filepath, 'w') as f:
            json.dump([asdict(log) for log in self.decision_history], f, indent=2)
        logger.info(f"Decision log exported to {filepath}")


class AdaptiveDecisionMaker(RealTimeDecisionMaker):
    """
    Enhanced decision maker that learns from outcomes
    Adjusts thresholds based on historical success rates
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_rate_by_action = {}
        self.false_positive_rate = 0.0
        
    def record_outcome(
        self,
        decision_timestamp: float,
        successful: bool,
        notes: str = ""
    ):
        """Record the outcome of a decision"""
        # Find the decision
        for log in self.decision_history:
            if abs(log.timestamp - decision_timestamp) < 1.0:
                log.outcome = "success" if successful else "failure"
                
                # Update success rates
                action = log.action_taken
                if action:
                    if action not in self.success_rate_by_action:
                        self.success_rate_by_action[action] = {'success': 0, 'total': 0}
                    
                    self.success_rate_by_action[action]['total'] += 1
                    if successful:
                        self.success_rate_by_action[action]['success'] += 1
                
                break
    
    def get_action_success_rate(self, action: str) -> float:
        """Get success rate for a specific action"""
        if action not in self.success_rate_by_action:
            return 0.5  # Neutral
        
        stats = self.success_rate_by_action[action]
        if stats['total'] == 0:
            return 0.5
        
        return stats['success'] / stats['total']
    
    def adjust_thresholds(self):
        """Automatically adjust decision thresholds based on outcomes"""
        # If too many false positives, increase urgency threshold
        total_decisions = len([log for log in self.decision_history if log.outcome])
        false_positives = len([
            log for log in self.decision_history
            if log.outcome == "failure" and log.urgency < 0.7
        ])
        
        if total_decisions > 50:
            self.false_positive_rate = false_positives / total_decisions
            
            if self.false_positive_rate > 0.3:
                # Too many false alarms, increase threshold
                self.action_urgency_threshold = min(0.8, self.action_urgency_threshold + 0.05)
                logger.info(f"Adjusted urgency threshold to {self.action_urgency_threshold:.2f}")
            elif self.false_positive_rate < 0.1:
                # Too conservative, decrease threshold
                self.action_urgency_threshold = max(0.3, self.action_urgency_threshold - 0.05)
                logger.info(f"Adjusted urgency threshold to {self.action_urgency_threshold:.2f}")
