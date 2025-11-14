import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import json
import pandas as pd
from src.data_ingestion.csv_parser import CSVParser
from src.models.drone_config import DroneConfig
from src.processing.stress_calculator import StressCalculator

def analyze_surveilldrone_net23():
    """Analyze SurveilDrone-Net23 dataset"""
    
    print("="*60)
    print("📹 ANALYZING: SurveilDrone-Net23")
    print("="*60)
    
    dataset_dir = Path(__file__).parent.parent / 'data' / 'public' / 'kaggle_datasets' / 'SurveilDrone-Net23'
    
    # Find all CSV files
    csv_files = list(dataset_dir.rglob('*.csv'))
    
    if not csv_files:
        print("\n❌ No CSV files found!")
        print(f"   Looking in: {dataset_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    results = []
    
    for i, csv_file in enumerate(csv_files[:10], 1):  # Analyze first 10
        print(f"\n[{i}/{min(len(csv_files), 10)}] {csv_file.name}")
        
        try:
            # Try to parse with our CSV parser
            parser = CSVParser(str(csv_file))
            records = parser.parse()
            
            if records:
                print(f"   ✅ Parsed {len(records)} records")
                
                # Get summary
                summary = parser.get_summary()
                print(f"   Duration: {summary.get('duration_seconds', 0):.1f} seconds")
                
                results.append({
                    'file': csv_file.name,
                    'records': len(records),
                    'duration': summary.get('duration_seconds', 0),
                    'success': True
                })
            else:
                print(f"   ⚠️  No records parsed")
                results.append({
                    'file': csv_file.name,
                    'success': False,
                    'error': 'No records'
                })
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            results.append({
                'file': csv_file.name,
                'success': False,
                'error': str(e)[:200]
            })
    
    # Summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n📊 Summary:")
    print(f"   Successful: {successful}/{len(results)}")
    
    return results

def analyze_uav_autonomous_navigation():
    """Analyze UAV Autonomous Navigation dataset"""
    
    print("\n" + "="*60)
    print("🚁 ANALYZING: UAV Autonomous Navigation")
    print("="*60)
    
    dataset_dir = Path(__file__).parent.parent / 'data' / 'public' / 'kaggle_datasets' / 'UAV-Autonomous-Navigation'
    
    # Find all data files
    csv_files = list(dataset_dir.rglob('*.csv'))
    log_files = list(dataset_dir.rglob('*.log'))
    
    print(f"\nFound:")
    print(f"   CSV files: {len(csv_files)}")
    print(f"   LOG files: {len(log_files)}")
    
    results = []
    
    # Analyze CSV files
    for i, csv_file in enumerate(csv_files[:10], 1):
        print(f"\n[{i}/{min(len(csv_files), 10)}] {csv_file.name}")
        
        try:
            parser = CSVParser(str(csv_file))
            records = parser.parse()
            
            if records:
                print(f"   ✅ Parsed {len(records)} records")
                
                summary = parser.get_summary()
                print(f"   Duration: {summary.get('duration_seconds', 0):.1f} seconds")
                
                results.append({
                    'file': csv_file.name,
                    'type': 'csv',
                    'records': len(records),
                    'success': True
                })
            else:
                print(f"   ⚠️  No records parsed")
                results.append({
                    'file': csv_file.name,
                    'type': 'csv',
                    'success': False
                })
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            results.append({
                'file': csv_file.name,
                'type': 'csv',
                'success': False,
                'error': str(e)[:200]
            })
    
    # Summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n📊 Summary:")
    print(f"   Successful: {successful}/{len(results)}")
    
    return results

def run_full_analysis():
    """Run analysis on both datasets"""
    
    print("="*60)
    print("🔬 FULL KAGGLE DATASET ANALYSIS")
    print("="*60)
    
    surveilldrone_results = analyze_surveilldrone_net23()
    uav_nav_results = analyze_uav_autonomous_navigation()
    
    # Combined summary
    all_results = surveilldrone_results + uav_nav_results
    total_successful = sum(1 for r in all_results if r.get('success'))
    
    print("\n" + "="*60)
    print("📈 OVERALL SUMMARY")
    print("="*60)
    print(f"\nTotal files analyzed: {len(all_results)}")
    print(f"Successfully parsed: {total_successful}")
    print(f"Failed: {len(all_results) - total_successful}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'data' / 'validation'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / 'kaggle_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'surveilldrone_net23': surveilldrone_results,
            'uav_autonomous_navigation': uav_nav_results,
            'summary': {
                'total': len(all_results),
                'successful': total_successful,
                'failed': len(all_results) - total_successful
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    run_full_analysis()