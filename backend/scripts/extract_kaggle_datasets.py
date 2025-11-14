import os
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import json

class KaggleDatasetExtractor:
    """Extract and organize manually downloaded Kaggle datasets"""
    
    def __init__(self):
        # FIXED PATH - go up 2 levels to project root
        self.base_dir = Path(__file__).parent.parent.parent / 'data'
        self.kaggle_dir = self.base_dir / 'public' / 'kaggle_datasets'
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        self.extracted_files = []
    
    def extract_surveilldrone_net23(self):
        """Extract and organize SurveilDrone-Net23 dataset"""
        
        print("="*60)
        print("📹 EXTRACTING: SurveilDrone-Net23")
        print("="*60)
        
        dataset_dir = self.kaggle_dir / 'SurveilDrone-Net23'
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\nDataset directory: {dataset_dir}")
        
        # Look for ZIP files
        zip_files = list(dataset_dir.glob('*.zip'))
        
        if zip_files:
            print(f"\nFound {len(zip_files)} ZIP files to extract:")
            for zip_file in zip_files:
                print(f"  📦 {zip_file.name}")
                
                try:
                    extract_dir = dataset_dir / zip_file.stem
                    extract_dir.mkdir(exist_ok=True)
                    
                    print(f"     Extracting to: {extract_dir.name}/")
                    
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    extracted = list(extract_dir.glob('*'))
                    print(f"     ✅ Extracted {len(extracted)} files")
                    
                    self.extracted_files.extend(extracted)
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
        
        # Scan for CSV/data files
        csv_files = list(dataset_dir.rglob('*.csv'))
        json_files = list(dataset_dir.rglob('*.json'))
        txt_files = list(dataset_dir.rglob('*.txt'))
        
        print(f"\n📊 Dataset Contents:")
        print(f"   CSV files: {len(csv_files)}")
        print(f"   JSON files: {len(json_files)}")
        print(f"   TXT files: {len(txt_files)}")
        
        if csv_files:
            print(f"\n📋 CSV Files Found:")
            for csv_file in csv_files[:10]:
                file_size = csv_file.stat().st_size / (1024 * 1024)
                print(f"   - {csv_file.name} ({file_size:.2f} MB)")
            
            if len(csv_files) > 10:
                print(f"   ... and {len(csv_files) - 10} more files")
        
        return {
            'name': 'SurveilDrone-Net23',
            'path': str(dataset_dir),
            'csv_files': [str(f) for f in csv_files],
            'json_files': [str(f) for f in json_files],
            'total_files': len(csv_files) + len(json_files) + len(txt_files)
        }
    
    def extract_uav_autonomous_navigation(self):
        """Extract and organize UAV Autonomous Navigation dataset"""
        
        print("\n" + "="*60)
        print("🚁 EXTRACTING: UAV Autonomous Navigation")
        print("="*60)
        
        dataset_dir = self.kaggle_dir / 'UAV-Autonomous-Navigation'
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\nDataset directory: {dataset_dir}")
        
        # Look for ZIP files
        zip_files = list(dataset_dir.glob('*.zip'))
        
        if zip_files:
            print(f"\nFound {len(zip_files)} ZIP files to extract:")
            for zip_file in zip_files:
                print(f"  📦 {zip_file.name}")
                
                try:
                    extract_dir = dataset_dir / zip_file.stem
                    extract_dir.mkdir(exist_ok=True)
                    
                    print(f"     Extracting to: {extract_dir.name}/")
                    
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    extracted = list(extract_dir.glob('*'))
                    print(f"     ✅ Extracted {len(extracted)} files")
                    
                    self.extracted_files.extend(extracted)
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
        
        # Scan for data files
        csv_files = list(dataset_dir.rglob('*.csv'))
        json_files = list(dataset_dir.rglob('*.json'))
        txt_files = list(dataset_dir.rglob('*.txt'))
        log_files = list(dataset_dir.rglob('*.log'))
        
        print(f"\n📊 Dataset Contents:")
        print(f"   CSV files: {len(csv_files)}")
        print(f"   JSON files: {len(json_files)}")
        print(f"   LOG files: {len(log_files)}")
        print(f"   TXT files: {len(txt_files)}")
        
        if csv_files:
            print(f"\n📋 CSV Files Found:")
            for csv_file in csv_files[:10]:
                file_size = csv_file.stat().st_size / (1024 * 1024)
                print(f"   - {csv_file.name} ({file_size:.2f} MB)")
            
            if len(csv_files) > 10:
                print(f"   ... and {len(csv_files) - 10} more files")
        
        return {
            'name': 'UAV-Autonomous-Navigation',
            'path': str(dataset_dir),
            'csv_files': [str(f) for f in csv_files],
            'json_files': [str(f) for f in json_files],
            'log_files': [str(f) for f in log_files],
            'total_files': len(csv_files) + len(json_files) + len(txt_files) + len(log_files)
        }
    
    def inspect_csv_structure(self, csv_file: Path, sample_rows: int = 5):
        """Inspect the structure of a CSV file"""
        
        print(f"\n🔍 Inspecting: {csv_file.name}")
        print("-" * 60)
        
        try:
            df = pd.read_csv(csv_file, nrows=sample_rows)
            
            print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            print(f"\n📋 Columns ({len(df.columns)}):")
            for i, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                non_null = df[col].notna().sum()
                print(f"   {i:2d}. {col:30s} ({dtype}, {non_null}/{len(df)} non-null)")
            
            print(f"\n📊 First {sample_rows} rows:")
            print(df.head(sample_rows).to_string())
            
            # Check for potential telemetry columns
            telemetry_keywords = [
                'gps', 'latitude', 'longitude', 'altitude', 'lat', 'lon',
                'speed', 'velocity', 'accel', 'gyro', 'imu',
                'pitch', 'roll', 'yaw', 'heading',
                'battery', 'voltage', 'current',
                'motor', 'rpm', 'throttle',
                'timestamp', 'time'
            ]
            
            found_keywords = []
            for col in df.columns:
                col_lower = col.lower()
                for keyword in telemetry_keywords:
                    if keyword in col_lower:
                        found_keywords.append(col)
                        break
            
            if found_keywords: 
                
                print(f"\n🎯 Potential telemetry columns found:")
                for col in found_keywords:
                    print(f"   ✓ {col}")
            
            return {
                'file': csv_file.name,
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'telemetry_columns': found_keywords
            }
            
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            return None
    
    def analyze_all_datasets(self):
        """Extract and analyze all datasets"""
        
        print("="*60)
        print("🔬 KAGGLE DATASET EXTRACTION & ANALYSIS")
        print("="*60)
        
        dataset_info = []
        
        # Extract both datasets
        surveilldrone_info = self.extract_surveilldrone_net23()
        dataset_info.append(surveilldrone_info)
        
        uav_nav_info = self.extract_uav_autonomous_navigation()
        dataset_info.append(uav_nav_info)
        
        # Inspect CSV files
        print("\n" + "="*60)
        print("📊 CSV FILE INSPECTION")
        print("="*60)
        
        all_csv_files = []
        for info in dataset_info:
            all_csv_files.extend([Path(f) for f in info.get('csv_files', [])])
        
        csv_inspections = []
        
        if all_csv_files:
            print(f"\nInspecting {len(all_csv_files)} CSV files...")
            
            for i, csv_file in enumerate(all_csv_files[:5], 1):
                print(f"\n[{i}/5]")
                inspection = self.inspect_csv_structure(csv_file)
                if inspection:
                    csv_inspections.append(inspection)
            
            if len(all_csv_files) > 5:
                print(f"\n... {len(all_csv_files) - 5} more CSV files not inspected")
        
        # Create comprehensive index
        index = {
            'datasets': dataset_info,
            'csv_inspections': csv_inspections,
            'total_csv_files': len(all_csv_files),
            'extraction_summary': {
                'total_files_extracted': len(self.extracted_files),
                'datasets_processed': len(dataset_info)
            }
        }
        
        # Save index
        index_file = self.kaggle_dir / 'extraction_index.json'
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print(f"\n📑 Index saved to: {index_file}")
        print(f"📊 Total CSV files: {len(all_csv_files)}")
        print(f"📦 Total datasets: {len(dataset_info)}")
        
        # Recommendations
        print("\n" + "="*60)
        print("🎯 NEXT STEPS")
        print("="*60)
        print("\n1. Review CSV structures above")
        print("2. Identify which files contain flight telemetry")
        print("3. Create custom parsers for each format")
        print("4. Run batch analysis on all files")
        
        return index

def main():
    extractor = KaggleDatasetExtractor()
    index = extractor.analyze_all_datasets()
    
    print("\n" + "="*60)
    print("📂 DATASET LOCATIONS")
    print("="*60)
    print(f"\nSurveilDrone-Net23:")
    print(f"  {extractor.kaggle_dir / 'SurveilDrone-Net23'}")
    print(f"\nUAV Autonomous Navigation:")
    print(f"  {extractor.kaggle_dir / 'UAV-Autonomous-Navigation'}")
    
    print("\n💡 To analyze these datasets:")
    print("   python analyze_kaggle_datasets.py")

if __name__ == "__main__":
    main()