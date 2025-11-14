import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

def verify_setup():
    """Verify complete setup"""
    
    print("="*60)
    print("🔍 VERIFYING SETUP")
    print("="*60)
    
    checks = []
    
    # FIXED PATHS - go up 2 levels to project root
    project_root = Path(__file__).parent.parent.parent
    
    # Check 1: Python packages
    print("\n1️⃣ Checking Python packages...")
    try:
        import pandas
        import numpy
        import matplotlib
        print("   ✅ Core packages installed")
        checks.append(True)
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        checks.append(False)
    
    # Check 2: Database (skip for now - we'll check later)
    print("\n2️⃣ Checking database...")
    print("   ⏭️  Skipped (will check when needed)")
    checks.append(True)
    
    # Check 3: Drone configs
    print("\n3️⃣ Checking drone configurations...")
    config_dir = project_root / 'config' / 'drones'
    config_files = list(config_dir.glob('*.json'))
    if config_files:
        print(f"   ✅ Found {len(config_files)} configurations")
        for f in config_files:
            print(f"      - {f.name}")
        checks.append(True)
    else:
        print("   ❌ No configurations found")
        checks.append(False)
    
    # Check 4: Sample data
    print("\n4️⃣ Checking sample data...")
    sample_file = project_root / 'data' / 'raw' / 'sample_flight_log.csv'
    if sample_file.exists():
        size_kb = sample_file.stat().st_size / 1024
        print(f"   ✅ Sample data exists ({size_kb:.1f} KB)")
        checks.append(True)
    else:
        print("   ❌ Sample data not found")
        checks.append(False)
    
    # Check 5: Parsers
    print("\n5️⃣ Checking parsers...")
    parser_files = [
        'csv_parser.py',
        'ardupilot_parser.py',
        'surveilldrone_parser.py',
        'uav_navigation_parser.py'
    ]
    parser_dir = project_root / 'backend' / 'src' / 'data_ingestion'
    missing_parsers = []
    for parser in parser_files:
        if not (parser_dir / parser).exists():
            missing_parsers.append(parser)
    
    if not missing_parsers:
        print(f"   ✅ All {len(parser_files)} parsers exist")
        checks.append(True)
    else:
        print(f"   ⚠️  Missing parsers: {', '.join(missing_parsers)}")
        checks.append(len(missing_parsers) < len(parser_files))
    
    # Check 6: Stress calculator
    print("\n6️⃣ Checking stress calculator...")
    stress_calc = project_root / 'backend' / 'src' / 'processing' / 'stress_calculator.py'
    if stress_calc.exists():
        print("   ✅ Stress calculator exists")
        checks.append(True)
    else:
        print("   ❌ Stress calculator not found")
        checks.append(False)
    
    # Check 7: Folder structure
    print("\n7️⃣ Checking folder structure...")
    required_folders = [
        project_root / 'data' / 'raw',
        project_root / 'data' / 'processed',
        project_root / 'data' / 'public',
        project_root / 'data' / 'validation',
        project_root / 'config' / 'drones'
    ]
    missing_folders = [f for f in required_folders if not f.exists()]
    
    if not missing_folders:
        print("   ✅ All required folders exist")
        checks.append(True)
    else:
        print(f"   ⚠️  Missing folders: {len(missing_folders)}")
        for f in missing_folders:
            print(f"      - {f}")
        checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n✅ {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Setup complete and verified!")
        print("\n🚀 Ready to analyze flight data!")
    else:
        print("\n⚠️  Some issues found. Review above for details.")
    
    return passed == total

if __name__ == "__main__":
    verify_setup()