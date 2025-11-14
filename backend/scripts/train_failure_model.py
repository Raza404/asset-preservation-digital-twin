# backend/scripts/train_failure_model.py

import sys
import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

def load_master_dataset(processed_dir):
    """Loads and combines all labeled CSVs from the processed directory."""
    print(f"Loading all labeled data from {processed_dir}...")
    
    csv_files = glob.glob(os.path.join(processed_dir, "labeled_*.csv"))
    if not csv_files:
        print("Error: No 'labeled_*.csv' files found in data/processed/")
        return None
        
    df_list = []
    for f in csv_files:
        print(f"   Loading {os.path.basename(f)}...")
        df_list.append(pd.read_csv(f))
        
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"\n✅ Master dataset loaded. Total records: {len(master_df)}")
    print("--- Data Balance ---")
    print(master_df['failure_type'].value_counts())
    
    return master_df

def main():
    print("============================================================")
    print("🤖 FAILURE PREDICTION MODEL - TRAINER")
    print("============================================================")

    # 1. Load and consolidate all labeled data
    processed_dir = os.path.join(project_root, 'data', 'processed')
    master_df = load_master_dataset(processed_dir)
    if master_df is None:
        return

    # 2. Define Features (X) and Target (y)
    print("\n2️⃣ Defining features (X) and target (y)...")
    
    y = master_df['failure_type']
    
    feature_cols = [
        col for col in master_df.columns if 
        'rolling' in col or 
        'magnitude' in col or 
        'g_force' in col or
        'vibration' in col or
        'jerk' in col or
        'rpm' in col or
        'current' in col
    ]
    
    if 'failure_type' in feature_cols:
        feature_cols.remove('failure_type')

    X = master_df[feature_cols]
    X = X.fillna(0)
    
    print(f"   Using {len(feature_cols)} features.")
    
    # 3. Split Data into Training and Testing sets
    print("\n3️⃣ Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        random_state=42,
        stratify=y  # Ensures all 3 classes are in both train and test
    )
    print(f"   Training records: {len(y_train)}")
    print(f"   Testing records:  {len(y_test)}")

    # 4. Train the Model
    print("\n4️⃣ Training RandomForestClassifier...")
    print("   This may take a moment...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced", # Handles imbalance
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 5. Evaluate the Model
    print("\n5️⃣ Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    print("\n--- TEST SET ACCURACY ---")
    print(f"   {accuracy_score(y_test, y_pred):.2%}")
    print("   (NOTE: Accuracy is misleading. Focus on the Recall scores below!)")
    
    # === FIX #1: Update target_names list to 3 items ===
    # We must also specify 'labels' to ensure the order is 0, 1, 2
    class_names = ['Type 0: Normal', 'Type 1: Motor Fail', 'Type 2: Vib Fail']
    class_labels = [0, 1, 2]
    # ===================================================
    
    print("\n--- CLASSIFICATION REPORT ---")
    # Add zero_division=0 to prevent warnings if a class has no predictions
    print(classification_report(y_test, y_pred, labels=class_labels, target_names=class_names, zero_division=0))
    
    print("\n--- CONFUSION MATRIX ---")
    print("Predicted ->")
    print("Actual")
    print("   v")
    
    # === FIX #2: Update confusion matrix labels ===
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=class_labels), 
                 index=['True: Normal', 'True: Motor', 'True: Vib'], 
                 columns=['Pred: Normal', 'Pred: Motor', 'Pred: Vib']))
    # ===============================================

    # --- STEP 5b: FEATURE IMPORTANCE ---
    print("\n5b: Analyzing Feature Importance...")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("\n--- TOP 15 MOST PREDICTIVE FEATURES ---")
    print(feature_importance_df.head(15).to_string())
    # ---------------------------------------------

    # 6. Save the Trained Model
    model_path = os.path.join(project_root, 'backend', 'src', 'ml', 'failure_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n\n✅ Model saved to: {model_path}")
    print("Multi-class model training complete!")

if __name__ == "__main__":
    main()