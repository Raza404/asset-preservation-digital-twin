# API Reference

Complete API documentation for the Digital Twin System.

## Table of Contents

1. [Digital Twin Engine](#digital-twin-engine)
2. [Anomaly Detection](#anomaly-detection)
3. [Path Planning](#path-planning)
4. [Telemetry Simulator](#telemetry-simulator)

---

## Digital Twin Engine

Main orchestration class for the digital twin system.

### Class: `DigitalTwinEngine`

**Constructor:**
```python
DigitalTwinEngine(contamination=0.1, safety_margin=10.0)
```

**Parameters:**
- `contamination` (float): Expected anomaly rate for ML model (0-0.5). Default: 0.1
- `safety_margin` (float): Safety margin for path planning in meters. Default: 10.0

**Methods:**

#### `initialize(training_data: np.ndarray) -> bool`
Initialize the system with training data.

**Parameters:**
- `training_data`: Historical telemetry data (N x 6 array)

**Returns:**
- `bool`: True if initialization successful

**Example:**
```python
engine = DigitalTwinEngine()
training_data = simulator.generate_training_dataset(1000)
engine.initialize(training_data)
```

#### `start_mission(start_position, end_position) -> dict`
Start a new mission with initial path planning.

**Parameters:**
- `start_position` (tuple): Starting position (x, y, z)
- `end_position` (tuple): Destination position (x, y, z)

**Returns:**
- `dict`: Mission info with status, positions, and path metrics

**Example:**
```python
mission = engine.start_mission((0, 0, 50), (100, 100, 50))
```

#### `update_telemetry(telemetry, current_position) -> dict`
Process new telemetry data and update system state.

**Parameters:**
- `telemetry` (np.ndarray): Current telemetry reading (6 features)
- `current_position` (tuple): Current drone position (x, y, z)

**Returns:**
- `dict`: System update with timestamp, position, risk assessment, and path update status

**Example:**
```python
update = engine.update_telemetry(telemetry, (10, 10, 50))
if update['path_update_required']:
    # Handle path replanning
```

#### `replan_trajectory(current_position, destination) -> dict`
Dynamically replan trajectory based on current risk.

**Parameters:**
- `current_position` (tuple): Current drone position (x, y, z)
- `destination` (tuple): Target destination (x, y, z)

**Returns:**
- `dict`: New trajectory with path and metrics

#### `get_system_status() -> dict`
Get comprehensive system status.

**Returns:**
- `dict`: Complete system state including risk summary

#### `stop_mission() -> dict`
Stop the current mission.

**Returns:**
- `dict`: Mission summary with statistics

---

## Anomaly Detection

ML-based anomaly detector for drone telemetry.

### Class: `AnomalyDetector`

**Constructor:**
```python
AnomalyDetector(contamination=0.1, n_estimators=100)
```

**Parameters:**
- `contamination` (float): Expected proportion of outliers (0-0.5). Default: 0.1
- `n_estimators` (int): Number of trees in Isolation Forest. Default: 100

**Telemetry Features (in order):**
1. `battery_level` - Battery charge percentage (0-100)
2. `temperature` - Temperature in Celsius
3. `vibration` - Vibration level (arbitrary units)
4. `altitude` - Altitude in meters
5. `speed` - Speed in m/s
6. `motor_current` - Motor current in Amperes

**Methods:**

#### `train(telemetry_data: np.ndarray) -> None`
Train the anomaly detection model.

**Parameters:**
- `telemetry_data` (np.ndarray): Historical telemetry data (N x 6)

**Example:**
```python
detector = AnomalyDetector()
detector.train(training_data)
```

#### `predict(telemetry_sample: np.ndarray) -> tuple`
Predict if a telemetry sample is anomalous.

**Parameters:**
- `telemetry_sample` (np.ndarray): Single telemetry reading (6 features)

**Returns:**
- `tuple`: (prediction, anomaly_score)
  - `prediction`: 1 for normal, -1 for anomaly
  - `anomaly_score`: Normalized score (0-1, higher = more anomalous)

**Example:**
```python
prediction, score = detector.predict(telemetry)
```

#### `detect_failure_risk(telemetry_sample: np.ndarray) -> dict`
Detect failure risk with detailed diagnostics.

**Parameters:**
- `telemetry_sample` (np.ndarray): Single telemetry reading (6 features)

**Returns:**
- `dict`: Risk assessment with keys:
  - `is_anomaly` (bool): Whether sample is anomalous
  - `anomaly_score` (float): Anomaly score (0-1)
  - `risk_level` (str): LOW, MEDIUM, HIGH, or CRITICAL
  - `recommendation` (str): Actionable recommendation

**Risk Level Thresholds:**
- `LOW`: anomaly_score < 0.3
- `MEDIUM`: 0.3 ≤ anomaly_score < 0.6
- `HIGH`: 0.6 ≤ anomaly_score < 0.8
- `CRITICAL`: anomaly_score ≥ 0.8

**Example:**
```python
risk = detector.detect_failure_risk(telemetry)
print(f"Risk: {risk['risk_level']}, Score: {risk['anomaly_score']}")
```

---

## Path Planning

Adaptive path planner for trajectory optimization.

### Class: `AdaptivePathPlanner`

**Constructor:**
```python
AdaptivePathPlanner(safety_margin=10.0)
```

**Parameters:**
- `safety_margin` (float): Safety margin in meters. Default: 10.0

**Methods:**

#### `plan_initial_path(start, end, obstacles=None) -> np.ndarray`
Plan initial optimal path from start to end.

**Parameters:**
- `start` (tuple): Starting position (x, y, z)
- `end` (tuple): End position (x, y, z)
- `obstacles` (list, optional): List of obstacles (not implemented)

**Returns:**
- `np.ndarray`: Array of waypoints (N x 3)

**Example:**
```python
planner = AdaptivePathPlanner()
path = planner.plan_initial_path((0, 0, 50), (100, 100, 50))
```

#### `optimize_trajectory(current_position, risk_assessment, destination) -> np.ndarray`
Dynamically optimize trajectory based on risk.

**Parameters:**
- `current_position` (tuple): Current position (x, y, z)
- `risk_assessment` (dict): Risk assessment from anomaly detector
- `destination` (tuple): Final destination (x, y, z)

**Returns:**
- `np.ndarray`: Optimized path (N x 3)

**Path Strategies by Risk Level:**
- `LOW`: Optimal direct path (20 waypoints)
- `MEDIUM`: Cautious path with more waypoints (30 waypoints)
- `HIGH`: Conservative path with reduced altitude (25 waypoints)
- `CRITICAL`: Emergency landing path (10 waypoints, descends to ground)

**Example:**
```python
risk = {'risk_level': 'HIGH'}
new_path = planner.optimize_trajectory((50, 50, 50), risk, (100, 100, 50))
```

#### `calculate_path_metrics(path: np.ndarray) -> dict`
Calculate metrics for a path.

**Parameters:**
- `path` (np.ndarray): Array of waypoints (N x 3)

**Returns:**
- `dict`: Metrics with keys:
  - `total_distance` (float): Total path length in meters
  - `total_altitude_change` (float): Sum of altitude changes
  - `smoothness` (float): Path smoothness (0-1, higher = smoother)
  - `num_waypoints` (int): Number of waypoints

**Example:**
```python
metrics = planner.calculate_path_metrics(path)
print(f"Distance: {metrics['total_distance']:.2f}m")
```

---

## Telemetry Simulator

Generates realistic drone telemetry for testing.

### Class: `TelemetrySimulator`

**Constructor:**
```python
TelemetrySimulator(seed=42)
```

**Parameters:**
- `seed` (int): Random seed for reproducibility. Default: 42

**Normal Operating Ranges:**
- Battery: 80-100%
- Temperature: 20-35°C
- Vibration: 0-2 units
- Altitude: 50-100m
- Speed: 5-15 m/s
- Motor Current: 2-5A

**Anomaly Ranges:**
- Battery: 10-40%
- Temperature: 45-70°C
- Vibration: 5-15 units
- Altitude: 5-30m
- Speed: 0.5-3 m/s
- Motor Current: 8-15A

**Methods:**

#### `generate_normal_telemetry(num_samples=1) -> np.ndarray`
Generate normal telemetry data.

**Parameters:**
- `num_samples` (int): Number of samples. Default: 1

**Returns:**
- `np.ndarray`: Telemetry samples (N x 6)

**Example:**
```python
simulator = TelemetrySimulator()
telemetry = simulator.generate_normal_telemetry(10)
```

#### `generate_anomalous_telemetry(num_samples=1, anomaly_type='random') -> np.ndarray`
Generate anomalous telemetry data.

**Parameters:**
- `num_samples` (int): Number of samples. Default: 1
- `anomaly_type` (str): Type of anomaly. Options: 'battery', 'temperature', 'vibration', 'random'

**Returns:**
- `np.ndarray`: Anomalous telemetry samples (N x 6)

**Example:**
```python
anomalous = simulator.generate_anomalous_telemetry(5, 'battery')
```

#### `generate_training_dataset(num_samples=1000, contamination=0.1) -> np.ndarray`
Generate training dataset with mixed normal and anomalous data.

**Parameters:**
- `num_samples` (int): Total number of samples. Default: 1000
- `contamination` (float): Proportion of anomalous samples. Default: 0.1

**Returns:**
- `np.ndarray`: Mixed training dataset (N x 6)

**Example:**
```python
training_data = simulator.generate_training_dataset(1000, contamination=0.1)
```

#### `simulate_flight_sequence(duration=20, anomaly_start=10) -> tuple`
Simulate a complete flight sequence.

**Parameters:**
- `duration` (int): Number of time steps. Default: 20
- `anomaly_start` (int): Time step when anomaly begins. Default: 10

**Returns:**
- `tuple`: (telemetry_sequence, positions)
  - `telemetry_sequence` (np.ndarray): Telemetry data (N x 6)
  - `positions` (list): Position tuples [(x, y, z), ...]

**Example:**
```python
telemetry_seq, positions = simulator.simulate_flight_sequence(duration=30)
```

#### `get_feature_names() -> list`
Get list of telemetry feature names.

**Returns:**
- `list`: Feature names

**Example:**
```python
features = simulator.get_feature_names()
# ['battery_level', 'temperature', 'vibration', 'altitude', 'speed', 'motor_current']
```

---

## Complete Usage Example

```python
from digital_twin_engine import DigitalTwinEngine
from telemetry_simulator import TelemetrySimulator

# 1. Initialize
simulator = TelemetrySimulator(seed=42)
engine = DigitalTwinEngine(contamination=0.1, safety_margin=10.0)

# 2. Train the system
training_data = simulator.generate_training_dataset(num_samples=1000)
engine.initialize(training_data)

# 3. Start mission
start = (0.0, 0.0, 50.0)
end = (100.0, 80.0, 50.0)
mission = engine.start_mission(start, end)

# 4. Real-time monitoring loop
telemetry_seq, positions = simulator.simulate_flight_sequence(duration=20)

for telemetry, position in zip(telemetry_seq, positions):
    # Process telemetry
    update = engine.update_telemetry(telemetry, position)
    
    # Check risk
    risk = update['risk_assessment']
    print(f"Position: {position}, Risk: {risk['risk_level']}")
    
    # Replan if needed
    if update['path_update_required']:
        result = engine.replan_trajectory(position, end)
        print(f"Path replanned: {result['path_metrics']['num_waypoints']} waypoints")

# 5. Get summary
status = engine.get_system_status()
summary = engine.stop_mission()

print(f"Mission completed. Total updates: {summary['total_telemetry_updates']}")
```

---

## Error Handling

Common errors and how to handle them:

### RuntimeError: "Model must be trained before prediction"
```python
# Solution: Train the model first
detector.train(training_data)
```

### RuntimeError: "Engine must be initialized before starting mission"
```python
# Solution: Initialize the engine
engine.initialize(training_data)
```

### ValueError: "Expected N features, got M"
```python
# Solution: Ensure telemetry has 6 features
telemetry = np.array([battery, temp, vibration, altitude, speed, current])
```

---

## Performance Considerations

- **Training Time**: ~1 second for 1000 samples
- **Prediction Time**: <10ms per sample
- **Path Planning Time**: <50ms per trajectory
- **Memory Usage**: <100MB for typical operations

## Best Practices

1. **Training Data**: Use at least 500 samples for training
2. **Contamination**: Set to expected anomaly rate (typically 0.05-0.15)
3. **Real-time Updates**: Process telemetry every 0.1-1 seconds
4. **Path Replanning**: Only replan when risk level changes significantly
5. **Safety Margin**: Adjust based on operational requirements (5-20 meters)
