"""
Train hybrid LSTM + Random Forest system for failure prediction.
This script trains both models and saves them for production use.
"""

import sys
import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add backend to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend'))

from src.ml.lstm_failure_predictor import LSTMFailurePredictor, TENSORFLOW_AVAILABLE


def load_and_prepare_data(processed_dir: str):
    """Load all labeled flight data."""
    print("📂 Loading labeled flight data...")
    
    csv_files = glob.glob(os.path.join(processed_dir, "labeled_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No labeled_*.csv files found in data/processed/")
    
    df_list = []
    for f in csv_files:
        print(f"   Loading {os.path.basename(f)}...")
        df_list.append(pd.read_csv(f))
    
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Loaded {len(master_df)} records from {len(csv_files)} files\n")
    
    return master_df


def prepare_features(df: pd.DataFrame):
    """Extract and prepare features for training."""
    print("🔧 Preparing features...")
    
    # Define feature columns
    feature_cols = [
        col for col in df.columns if any(keyword in col.lower() for keyword in [
            'rolling', 'magnitude', 'g_force', 'vibration', 'jerk',
            'rpm', 'current', 'temp', 'voltage', 'accel', 'gyro',
            'throttle', 'battery', 'motor', 'esc'
        ])
    ]
    
    # Remove target columns
    feature_cols = [col for col in feature_cols if 'failure' not in col.lower()]
    
    X = df[feature_cols].fillna(0)
    
    print(f"✅ Selected {len(feature_cols)} features\n")
    
    return X, feature_cols


def prepare_labels(df: pd.DataFrame):
    """Prepare labels for multi-label classification."""
    print("🏷️  Preparing labels...")
    
    # Create component-specific labels
    labels = pd.DataFrame()
    
    # Derive component failures from failure_type
    if 'failure_type' in df.columns:
        failure_types = df['failure_type'].fillna(0).astype(int)
        
        # Multi-label encoding
        labels['motor_failure'] = (failure_types == 1).astype(int)
        labels['vibration_failure'] = (failure_types == 2).astype(int)
        labels['battery_failure'] = ((failure_types == 1) | (df.get('battery_voltage', 12) < 10.5)).astype(int)
        labels['esc_failure'] = (failure_types == 1).astype(int)  # Often related to motor
        labels['sensor_failure'] = ((failure_types == 2) | (df.get('gps_satellites', 12) < 6)).astype(int)
    else:
        # Fallback: all healthy
        labels['motor_failure'] = 0
        labels['vibration_failure'] = 0
        labels['battery_failure'] = 0
        labels['esc_failure'] = 0
        labels['sensor_failure'] = 0
    
    print(f"✅ Label distribution:")
    print(labels.sum())
    print()
    
    return labels


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train enhanced Random Forest model."""
    print("\n" + "="*80)
    print("🌲 TRAINING RANDOM FOREST")
    print("="*80 + "\n")
    
    # For multi-label, train on failure_type if available
    if 'failure_type' in y_train.columns:
        y_train_rf = y_train['failure_type']
        y_test_rf = y_test['failure_type']
    else:
        # Use any failure as binary classification
        y_train_rf = (y_train.sum(axis=1) > 0).astype(int)
        y_test_rf = (y_test.sum(axis=1) > 0).astype(int)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train_rf)
    
    # Evaluate
    train_score = model.score(X_train, y_train_rf)
    test_score = model.score(X_test, y_test_rf)
    
    print(f"✅ Training complete!")
    print(f"   Train accuracy: {train_score*100:.2f}%")
    print(f"   Test accuracy: {test_score*100:.2f}%\n")
    
    # Feature importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    
    print("📊 Top 10 Most Important Features:")
    for i, idx in enumerate(top_indices, 1):
        if hasattr(model, 'feature_names_in_'):
            feat_name = model.feature_names_in_[idx]
        else:
            feat_name = f"Feature {idx}"
        print(f"   {i:2d}. {feat_name:40s} {importances[idx]:.4f}")
    
    return model


def train_lstm(X_train, y_train, X_val, y_val, sequence_length=30):
    """Train LSTM model for time-series prediction."""
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠️  TensorFlow not available. Skipping LSTM training.")
        return None
    
    print("\n" + "="*80)
    print("🧠 TRAINING LSTM")
    print("="*80 + "\n")
    
    # Prepare sequences
    print(f"Preparing sequences (length={sequence_length})...")
    predictor = LSTMFailurePredictor(sequence_length=sequence_length)
    
    # Convert to sequences
    X_train_seq = []
    y_train_seq = []
    
    for i in range(len(X_train) - sequence_length + 1):
        X_train_seq.append(X_train.iloc[i:i+sequence_length].values)
        y_train_seq.append(y_train.iloc[i+sequence_length-1].values)
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    
    # Validation sequences
    X_val_seq = []
    y_val_seq = []
    
    for i in range(len(X_val) - sequence_length + 1):
        X_val_seq.append(X_val.iloc[i:i+sequence_length].values)
        y_val_seq.append(y_val.iloc[i+sequence_length-1].values)
    
    X_val_seq = np.array(X_val_seq)
    y_val_seq = np.array(y_val_seq)
    
    print(f"✅ Training sequences: {X_train_seq.shape}")
    print(f"✅ Validation sequences: {X_val_seq.shape}\n")
    
    # Train
    print("Training LSTM (this may take several minutes)...")
    history = predictor.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=30,
        batch_size=64
    )
    
    print(f"\n✅ LSTM training complete!")
    print(f"   Final training loss: {history['loss'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return predictor


def main():
    """Main training workflow."""
    print("\n" + "="*80)
    print("🚀 HYBRID MODEL TRAINING PIPELINE")
    print("="*80)
    print("\nThis will train:")
    print("  1. Enhanced Random Forest (fast detection)")
    print("  2. LSTM Model (time-series prediction)")
    print()
    
    # Paths
    processed_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'backend', 'src', 'ml')
    
    # Load data
    df = load_and_prepare_data(processed_dir)
    
    # Prepare features and labels
    X, feature_cols = prepare_features(df)
    y = prepare_labels(df)
    
    # Add failure_type to labels if exists
    if 'failure_type' in df.columns:
        y['failure_type'] = df['failure_type'].fillna(0).astype(int)
    
    # Split data
    print("📊 Splitting data (70% train, 15% val, 15% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples\n")
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Save Random Forest
    rf_path = os.path.join(model_dir, 'failure_model_enhanced.joblib')
    joblib.dump(rf_model, rf_path)
    print(f"\n💾 Random Forest saved to: {rf_path}")
    
    # Train LSTM
    lstm_model = train_lstm(X_train, y_train, X_val, y_val)
    
    if lstm_model:
        # Save LSTM
        lstm_path = os.path.join(model_dir, 'lstm_failure_model.h5')
        lstm_model.save_model(lstm_path)
        print(f"💾 LSTM saved to: {lstm_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nModels are ready for use in the hybrid decision engine.")
    print("Run 'python scripts/demo_realtime_optimizer.py' to see them in action!\n")


if __name__ == "__main__":
    main()
