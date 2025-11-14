import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.betaflight_parser import BetaflightParser
from src.data_ingestion.csv_parser import CSVParser
from src.data_ingestion.ardupilot_parser import ArduPilotParser
from src.data_ingestion.px4_parser import PX4Parser
from src.data_ingestion.surveilldrone_parser import SurveilDroneParser
from src.data_ingestion.uav_navigation_parser import UAVNavigationParser
from pathlib import Path

def test_parser(file_path: str):
    """Test parser on a log file"""
    
    print("\n" + "="*60)
    print("LOG PARSER TEST")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        return
    
    file_name = Path(file_path).name.lower()
    ext = os.path.splitext(file_path)[1].lower()
    
    print(f"\nFile: {file_name}")
    print(f"Type: {ext}")
    
    # Determine parser based on filename or extension
    if 'surveilldrone' in file_name:
        print("Using SurveilDrone parser...")
        parser = SurveilDroneParser(file_path)

    elif 'betaflight' in file_name or 'my_flight' in file_name:
        print("Using Betaflight parser...")
        parser = BetaflightParser(file_path)
    
    elif 'uav_navigation' in file_name or 'uav-navigation' in file_name:
        print("Using UAV Navigation parser...")
        parser = UAVNavigationParser(file_path)
    elif ext == '.csv':
        print("Using CSV parser...")
        parser = CSVParser(file_path)
    elif ext == '.bin':
        print("Using ArduPilot parser...")
        parser = ArduPilotParser(file_path)
    elif ext == '.ulg':
        print("Using PX4 ULog parser...")
        parser = PX4Parser(file_path)
    else:
        print(f"\n⚠️  Unknown file type: {ext}")
        print("Trying CSV parser...")
        parser = CSVParser(file_path)
  
    # Parse the file
    records = parser.parse()
    
    if not records:
        print("\n❌ No records parsed!")
        return
    
    # Show summary
    print(f"\n📊 Summary:")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Show first few records
    df = parser.to_dataframe()
    print(f"\n📋 First 5 records:")
    available_cols = [col for col in ['timestamp', 'latitude', 'longitude', 'altitude', 'accel_x', 'accel_y', 'accel_z'] if col in df.columns]
    if available_cols:
        print(df[available_cols].head())
    else:
        print(df.head())
    
    # Show columns with data
    print(f"\n📝 Key telemetry columns:")
    key_cols = ['timestamp', 'latitude', 'longitude', 'altitude', 'ground_speed',
                'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                'battery_voltage', 'battery_remaining', 'wind_speed']
    
    for col in key_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                print(f"   ✓ {col:20s} ({non_null}/{len(df)} values)")
    
    # Save to processed
    output_path = os.path.join('..', '..', 'data', 'processed', f'parsed_{Path(file_path).stem}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    parser.save_to_csv(output_path)
    
    print(f"\n💾 Saved to: {output_path}")
    print("\n✅ Test complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Please provide a log file path")
        print("\nUsage:")
        print("   python test_parser.py <path_to_log_file>")
        print("\nExamples:")
        print("   python test_parser.py \"..\\..\\data\\public\\kaggle_datasets\\SurveilDrone-Net23\\SurveilDrone-Net23.csv\"")
        print("   python test_parser.py \"..\\..\\data\\public\\kaggle_datasets\\UAV-Autonomous-Navigation\\uav_navigation_dataset.csv\"")
        sys.exit(1)
    
    file_path = sys.argv[1]
    test_parser(file_path)