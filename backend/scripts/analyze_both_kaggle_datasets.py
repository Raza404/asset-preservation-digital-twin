import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from validate_with_public_data import ValidationRunner
import pandas as pd
import json

def analyze_both_datasets():
    """Analyze both Kaggle datasets with all drone configs"""
    
    print("="*60)
    print("🔬 COMPLETE KAGGLE DATASET ANALYSIS")
    print("="*60)
    
    # FIXED PATHS - go up 2 levels to project root
    kaggle_dir = Path(__file__).parent.parent.parent / 'data' / 'public' / 'kaggle_datasets'
    
    datasets = [
        kaggle_dir / 'SurveilDrone-Net23' / 'SurveilDrone-Net23.csv',
        kaggle_dir / 'UAV-Autonomous-Navigation' / 'uav_navigation_dataset.csv'
    ]
    
    # Drone configs
    config_dir = Path(__file__).parent.parent.parent / 'config' / 'drones'
    configs = list(config_dir.glob('*.json'))
    
    if not configs:
        print("❌ No drone configs found!")
        return
    
    print(f"\n📊 Datasets: {len(datasets)}")
    print(f"🔧 Configs: {len(configs)}")
    print(f"📈 Total analyses: {len(datasets) * len(configs)}")
    
    all_results = []
    
    # Analyze each dataset with each config
    for i, dataset_path in enumerate(datasets, 1):
        if not dataset_path.exists():
            print(f"\n⚠️  Skipping {dataset_path.name} - file not found")
            continue
        
        dataset_name = dataset_path.stem
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(datasets)}] {dataset_name}")
        print('='*60)
        
        for j, config_path in enumerate(configs, 1):
            config_name = config_path.stem
            
            print(f"\n   [{j}/{len(configs)}] Config: {config_name}")
            
            try:
                # Run validation
                validator = ValidationRunner(str(dataset_path), str(config_path))
                validator.run_validation()
                
                # Extract metrics
                passed = sum(1 for t in validator.validation_tests if t['passed'])
                total = len(validator.validation_tests)
                
                import numpy as np
                result = {
                    'dataset': dataset_name,
                    'config': config_name,
                    'tests_passed': passed,
                    'tests_total': total,
                    'success_rate': (passed/total*100) if total > 0 else 0,
                    'num_records': len(validator.records),
                    'avg_stress_score': float(np.mean(validator.stress_scores)),
                    'max_stress_score': float(max(validator.stress_scores)),
                    'avg_g_force': float(np.mean(validator.g_forces)),
                    'max_g_force': float(max(validator.g_forces)),
                    'high_stress_events': len(validator.high_stress_events)
                }
                
                all_results.append(result)
                
                print(f"      ✅ {passed}/{total} tests passed")
                print(f"      📊 Avg stress: {result['avg_stress_score']:.1f}/100")
                print(f"      🎢 Max G-force: {result['max_g_force']:.2f}G")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:150]}")
                all_results.append({
                    'dataset': dataset_name,
                    'config': config_name,
                    'error': str(e)[:300]
                })
    
    # Save results - FIXED PATH
    results_dir = Path(__file__).parent.parent.parent / 'data' / 'validation'
    results_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = results_dir / 'kaggle_datasets_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV
    csv_file = results_dir / 'kaggle_datasets_analysis.csv'
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 ANALYSIS SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in all_results if 'error' not in r)
    
    print(f"\n📊 Overall:")
    print(f"   Total analyses: {len(all_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(all_results) - successful}")
    
    if successful > 0:
        success_results = [r for r in all_results if 'error' not in r]
        
        print(f"\n🎯 Performance Metrics (Successful runs):")
        df_success = pd.DataFrame(success_results)
        print(f"   Avg stress score: {df_success['avg_stress_score'].mean():.1f}/100")
        print(f"   Avg G-force: {df_success['avg_g_force'].mean():.2f}G")
        print(f"   Total high-stress events: {df_success['high_stress_events'].sum()}")
        
        print(f"\n📂 By Dataset:")
        dataset_stats = df_success.groupby('dataset').agg({
            'avg_stress_score': 'mean',
            'max_g_force': 'max',
            'num_records': 'first'
        }).round(2)
        print(dataset_stats)
    
    print(f"\n💾 Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   CSV:  {csv_file}")
    
    print(f"\n📊 Visualizations saved in:")
    print(f"   {results_dir}")

if __name__ == "__main__":
    analyze_both_datasets()