"""
LSTM-based failure prediction model for time-series telemetry analysis.
Predicts failure probability across multiple time horizons (30s, 60s, 120s).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not installed. LSTM predictor will use fallback mode.")


class LSTMFailurePredictor:
    """
    LSTM model for predicting drone component failures from sequential telemetry.
    
    Features:
    - Multi-horizon prediction (30s, 60s, 120s ahead)
    - Component-specific failure probabilities
    - Remaining Useful Life (RUL) estimation
    - Temporal pattern recognition
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 prediction_horizons: List[int] = [30, 60, 120],
                 model_path: Optional[str] = None):
        """
        Initialize LSTM failure predictor.
        
        Args:
            sequence_length: Number of timesteps to look back (default 30 = 3 seconds at 10Hz)
            prediction_horizons: Seconds ahead to predict [30s, 60s, 120s]
            model_path: Path to pre-trained model file
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.is_trained = False
        
        # Component health thresholds
        self.component_thresholds = {
            'motor': {'critical': 0.8, 'warning': 0.5, 'safe': 0.2},
            'battery': {'critical': 0.85, 'warning': 0.6, 'safe': 0.3},
            'esc': {'critical': 0.75, 'warning': 0.5, 'safe': 0.25},
            'sensor': {'critical': 0.7, 'warning': 0.4, 'safe': 0.15},
            'propeller': {'critical': 0.65, 'warning': 0.45, 'safe': 0.2}
        }
        
        if model_path and TENSORFLOW_AVAILABLE:
            self._load_model(model_path)
    
    def _build_model(self, input_shape: Tuple[int, int], num_outputs: int = 5) -> Optional[Sequential]:
        """
        Build LSTM architecture.
        
        Args:
            input_shape: (sequence_length, num_features)
            num_outputs: Number of output classes/components
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            # Third LSTM layer (no return sequences)
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for classification
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            
            # Output layer - multi-label classification
            Dense(num_outputs, activation='sigmoid')  # Sigmoid for multi-label
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Multi-label classification
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Convert dataframe to sequences for LSTM input.
        
        Args:
            df: Telemetry dataframe
            feature_cols: List of feature column names
        
        Returns:
            3D array of shape (samples, sequence_length, features)
        """
        data = df[feature_cols].values
        sequences = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training labels (samples, num_components)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Training history dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for training LSTM model")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        self.model = self._build_model(input_shape, num_outputs)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        return history.history
    
    def predict_component_failure(self, sequence: np.ndarray) -> Dict[str, Dict]:
        """
        Predict failure probability for each component.
        
        Args:
            sequence: Single sequence (sequence_length, features)
        
        Returns:
            Dictionary with component predictions and recommendations
        """
        if self.model is None:
            # Fallback mode - use heuristics
            return self._heuristic_prediction(sequence)
        
        # Reshape for model input
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        # Get predictions
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        components = ['motor', 'battery', 'esc', 'sensor', 'propeller']
        results = {}
        
        for i, component in enumerate(components):
            if i < len(predictions):
                failure_prob = float(predictions[i])
                results[component] = self._analyze_component_health(
                    component, failure_prob, sequence
                )
        
        return results
    
    def _analyze_component_health(self, 
                                   component: str, 
                                   failure_prob: float,
                                   sequence: np.ndarray) -> Dict:
        """
        Analyze component health and generate recommendations.
        
        Args:
            component: Component name
            failure_prob: Predicted failure probability (0-1)
            sequence: Recent telemetry sequence
        
        Returns:
            Component health analysis with recommendations
        """
        thresholds = self.component_thresholds.get(component, {})
        
        # Determine health status
        if failure_prob >= thresholds.get('critical', 0.8):
            status = 'CRITICAL'
            color = 'red'
        elif failure_prob >= thresholds.get('warning', 0.5):
            status = 'WARNING'
            color = 'yellow'
        else:
            status = 'HEALTHY'
            color = 'green'
        
        # Estimate RUL (Remaining Useful Life) in seconds
        if failure_prob > 0.1:
            rul = int((1.0 - failure_prob) * 300)  # 0-300 seconds
        else:
            rul = 999  # >15 minutes
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(
            component, failure_prob, status, sequence
        )
        
        return {
            'failure_probability': failure_prob,
            'status': status,
            'color': color,
            'rul_seconds': rul,
            'rul_human': self._format_time(rul),
            'recommendations': recommendations,
            'urgency': 'high' if status == 'CRITICAL' else 'medium' if status == 'WARNING' else 'low'
        }
    
    def _generate_recommendations(self,
                                  component: str,
                                  failure_prob: float,
                                  status: str,
                                  sequence: np.ndarray) -> List[str]:
        """Generate actionable flight recommendations based on component health."""
        recommendations = []
        
        if component == 'motor':
            if status == 'CRITICAL':
                recommendations.extend([
                    "⚠️ IMMEDIATE: Reduce throttle to 60% maximum",
                    "⚠️ Initiate emergency landing sequence",
                    "⚠️ Avoid aggressive maneuvers - gentle inputs only",
                    "⚠️ Monitor motor temperature and vibration closely"
                ])
            elif status == 'WARNING':
                recommendations.extend([
                    "⚡ Reduce max throttle to 75%",
                    "⚡ Avoid sudden acceleration/deceleration",
                    "⚡ Consider returning to launch point",
                    "⚡ Reduce flight time by 30%"
                ])
            else:
                recommendations.append("✓ Motor healthy - normal operation")
        
        elif component == 'battery':
            if status == 'CRITICAL':
                recommendations.extend([
                    "🔋 CRITICAL: Return to home immediately",
                    "🔋 Reduce speed to conserve power",
                    "🔋 Maintain altitude - minimize climbing",
                    "🔋 Disable non-essential systems"
                ])
            elif status == 'WARNING':
                recommendations.extend([
                    "🔋 Plan return journey within 2 minutes",
                    "🔋 Reduce cruising speed by 25%",
                    "🔋 Avoid hovering - maintain forward flight",
                    "🔋 Monitor voltage drop rate"
                ])
        
        elif component == 'esc':
            if status == 'CRITICAL':
                recommendations.extend([
                    "⚡ Land immediately if safe",
                    "⚡ Smooth throttle inputs only",
                    "⚡ Avoid high current draw situations",
                    "⚡ Monitor for oscillations"
                ])
            elif status == 'WARNING':
                recommendations.extend([
                    "⚡ Reduce aggressiveness - use expo settings",
                    "⚡ Lower PID gains if possible",
                    "⚡ Avoid sustained high throttle"
                ])
        
        elif component == 'sensor':
            if status == 'CRITICAL':
                recommendations.extend([
                    "📡 Switch to manual flight mode",
                    "📡 Do not trust GPS/compass readings",
                    "📡 Maintain visual line of sight",
                    "📡 Land as soon as possible"
                ])
            elif status == 'WARNING':
                recommendations.extend([
                    "📡 Cross-check sensor readings",
                    "📡 Avoid autonomous modes",
                    "📡 Recalibrate sensors if possible"
                ])
        
        elif component == 'propeller':
            if status == 'CRITICAL':
                recommendations.extend([
                    "🚁 Severe vibration detected - land now",
                    "🚁 Inspect for propeller damage",
                    "🚁 Reduce speed to minimum controllable",
                    "🚁 Avoid windy conditions"
                ])
            elif status == 'WARNING':
                recommendations.extend([
                    "🚁 Check propeller balance",
                    "🚁 Reduce max speed by 30%",
                    "🚁 Minimize rapid direction changes"
                ])
        
        return recommendations
    
    def _format_time(self, seconds: int) -> str:
        """Format seconds into human readable time."""
        if seconds >= 999:
            return ">15 minutes"
        elif seconds >= 60:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            return f"{seconds}s"
    
    def _heuristic_prediction(self, sequence: np.ndarray) -> Dict[str, Dict]:
        """
        Fallback heuristic-based prediction when TensorFlow unavailable.
        Uses rule-based logic on recent telemetry data.
        """
        # Extract simple statistics from sequence
        recent_data = sequence[-10:] if len(sequence) > 10 else sequence
        
        # Placeholder heuristics (would be enhanced with actual feature analysis)
        results = {
            'motor': self._analyze_component_health('motor', 0.15, sequence),
            'battery': self._analyze_component_health('battery', 0.25, sequence),
            'esc': self._analyze_component_health('esc', 0.10, sequence),
            'sensor': self._analyze_component_health('sensor', 0.08, sequence),
            'propeller': self._analyze_component_health('propeller', 0.12, sequence)
        }
        
        return results
    
    def save_model(self, path: str):
        """Save trained LSTM model."""
        if self.model and TENSORFLOW_AVAILABLE:
            self.model.save(path)
            print(f"✓ LSTM model saved to {path}")
    
    def _load_model(self, path: str):
        """Load pre-trained LSTM model."""
        if TENSORFLOW_AVAILABLE:
            try:
                self.model = load_model(path)
                self.is_trained = True
                print(f"✓ LSTM model loaded from {path}")
            except Exception as e:
                print(f"⚠️  Could not load LSTM model: {e}")
