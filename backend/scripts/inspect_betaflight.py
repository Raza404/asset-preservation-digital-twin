import pandas as pd
import sys
from pathlib import Path

def find_data_start(filepath):
    """Find where actual flight data starts in Betaflight CSV"""
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # Look for the header line with column names
            if 'loopIteration' in line or 'time (us)' in line.lower() or 'gyroADC' in line:
                return i
    return 0

def inspect_betaflight_log(filepath):
    """Inspect Betaflight CSV structure"""
    
    print("="*60)
    print("🔍 BETAFLIGHT LOG INSPECTION")
    print("="*60)
    
    try:
        # Find where data starts
        data_start_row = find_data_start(filepath)
        
        print(f"\n📍 Data starts at row: {data_start_row}")
        
        if data_start_row > 0:
            print(f"   (Skipping {data_start_row} header/metadata rows)")
        
        # Read actual flight data
        df = pd.read_csv(filepath, skiprows=data_start_row, nrows=100)
        
        # Check if we got actual data
        if len(df.columns) < 10:
            print("\n⚠️  Still looks like metadata. Let me check the file structure...")
            
            # Show first 20 lines raw
            with open(filepath, 'r') as f:
                lines = [next(f) for _ in range(min(20, sum(1 for _ in open(filepath))))]
            
            print("\n📄 FIRST 20 LINES OF FILE:")
            for i, line in enumerate(lines):
                print(f"   {i:3d}: {line.strip()[:100]}")
            
            print("\n💡 Please tell me which line number has the column headers!")
            print("   (Look for line with: time, gyro, accel, motor, etc.)")
            return None
        
        file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n📁 File: {Path(filepath).name}")
        print(f"📦 Size: {file_size:.2f} MB")
        print(f"📏 Shape: {len(df)} rows × {len(df.columns)} columns (showing first 100)")
        
        print(f"\n📋 ALL COLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            sample = df[col].iloc[0] if len(df) > 0 else None
            print(f"   {i:3d}. {col:40s} | {str(dtype):10s} | Sample: {sample}")
        
        print(f"\n📊 FIRST 5 ROWS:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head())
        
        print(f"\n🎯 BETAFLIGHT TELEMETRY DETECTION:")
        
        # Common Betaflight columns
        betaflight_keywords = {
            'Time': ['time', 'loopiteration', 'timestamp'],
            'Gyroscope': ['gyroadu', 'gyro', 'gyrodata'],
            'Accelerometer': ['accsmooth', 'accel', 'accadu'],
            'Motors': ['motor[0]', 'motor[1]', 'motor[2]', 'motor[3]', 'motor'],
            'RC Commands': ['rccommand', 'setpoint'],
            'Attitude': ['attitude', 'roll', 'pitch', 'yaw'],
            'Battery': ['vbat', 'voltage', 'amperage', 'current', 'mah'],
            'GPS': ['gps', 'latitude', 'longitude', 'altitude'],
            'Barometer': ['baro'],
            'PID': ['axiserror', 'pid']
        }
        
        found_categories = {}
        for category, keywords in betaflight_keywords.items():
            found = []
            for col in df.columns:
                col_lower = col.lower()
                for keyword in keywords:
                    if keyword.lower() in col_lower:
                        found.append(col)
                        break
            if found:
                found_categories[category] = found
        
        if found_categories:
            for category, cols in found_categories.items():
                print(f"\n   {category}:")
                for col in cols:
                    non_null = df[col].notna().sum()
                    print(f"      ✓ {col:40s} ({non_null}/{len(df)} values)")
        else:
            print("\n   ⚠️  No standard Betaflight columns detected")
        
        # Check for data
        print(f"\n📈 DATA QUALITY:")
        total_cols = len(df.columns)
        cols_with_data = sum(1 for col in df.columns if df[col].notna().any())
        print(f"   Columns with data: {cols_with_data}/{total_cols}")
        
        if cols_with_data > 10:
            print(f"   ✅ Looks good - ready to parse!")
        
        return {
            'skip_rows': data_start_row,
            'columns': df.columns.tolist(),
            'found_categories': found_categories
        }
        
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        
        # Show raw file structure
        print("\n📄 Showing raw file structure...")
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    print(f"   Line {i}: {line.strip()[:100]}")
        except:
            pass
        
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Please provide Betaflight log file path")
        print("\nUsage:")
        print("   python inspect_betaflight.py <path_to_csv>")
        print("\nExample:")
        print("   python inspect_betaflight.py \"..\\..\\data\\public\\betaflight_logs\\my_flight_001.csv\"")
        sys.exit(1)
    
    filepath = sys.argv[1]
    result = inspect_betaflight_log(filepath)
    
    if result:
        print("\n" + "="*60)
        print("✅ INSPECTION COMPLETE")
        print("="*60)
        print(f"\nSkip {result['skip_rows']} rows when parsing")
        print(f"Found {len(result['columns'])} data columns")