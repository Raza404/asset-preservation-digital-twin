import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from src.data_ingestion.csv_parser import CSVParser
from src.data_ingestion.surveilldrone_parser import SurveilDroneParser
from src.data_ingestion.uav_navigation_parser import UAVNavigationParser
from src.data_ingestion.betaflight_parser import BetaflightParser
from src.models.drone_config import DroneConfig
from src.processing.stress_calculator import StressCalculator

class ValidationRunner:
    """Run validation tests on public flight data"""
    
    def __init__(self, log_file: str, config_file: str):
        self.log_file = log_file
        self.config_file = config_file
        self.records = []
        self.config = None
        self.stress_scores = []
        self.g_forces = []
        self.high_stress_events = []
        self.validation_tests = []
        self.results = {}
    
    def load_data(self):
        """Load and parse flight log"""
        
        print("\n" + "="*60)
        print("📖 LOADING FLIGHT DATA")
        print("="*60)
        
        file_name = Path(self.log_file).name.lower()
        
        # Choose appropriate parser
        if 'surveilldrone' in file_name:
            print("Using SurveilDrone parser...")
            parser = SurveilDroneParser(self.log_file)
        elif 'uav_navigation' in file_name:
            print("Using UAV Navigation parser...")
            parser = UAVNavigationParser(self.log_file)
        elif 'betaflight' in file_name or 'my_flight' in file_name:
            print("Using Betaflight parser...")
            parser = BetaflightParser(self.log_file)
        else:
            print("Using CSV parser...")
            parser = CSVParser(self.log_file)
        
        self.records = parser.parse()
        
        if not self.records:
            raise ValueError("No records parsed from log file")
        
        print(f"✅ Loaded {len(self.records)} telemetry records")
    
    def load_config(self):
        """Load drone configuration"""
        
        print("\n🔧 Loading drone configuration...")
        self.config = DroneConfig.load_from_file(self.config_file)
        print(f"✅ Loaded: {self.config.model_name}")
    
    def calculate_stress(self):
        """Calculate stress scores for all records"""
        
        print("\n⚙️  Calculating stress scores...")
        
        calculator = StressCalculator(self.config)
        
        for i, record in enumerate(self.records):
            # Calculate G-force from accelerometer data
            if record.accel_x is not None and record.accel_y is not None and record.accel_z is not None:
                # Calculate total acceleration magnitude
                accel_magnitude = np.sqrt(record.accel_x**2 + record.accel_y**2 + record.accel_z**2)
                # Convert to G-force (divide by gravity)
                g_force = accel_magnitude / 9.81
            else:
                g_force = 1.0
            
            self.g_forces.append(g_force)
            
            # Get environmental data
            wind_speed = record.wind_speed if record.wind_speed is not None else 0.0
            air_temp = record.air_temperature if record.air_temperature is not None else 25.0
            altitude = record.altitude if record.altitude is not None else 0.0
            
            # Calculate stress
            stress = calculator.calculate_flight_stress(
                g_force=g_force,
                wind_speed=wind_speed,
                air_temperature=air_temp,
                altitude=altitude
            )
            
            self.stress_scores.append(stress['overall_stress'])
            
            # Track high stress events (>70% stress)
            if stress['overall_stress'] > 70:
                self.high_stress_events.append({
                    'timestamp': record.timestamp,
                    'stress': stress['overall_stress'],
                    'g_force': g_force,
                    'components': stress['components']
                })
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1}/{len(self.records)} records...")
        
        print(f"✅ Calculated stress for {len(self.records)} records")
        print(f"   Average G-force: {np.mean(self.g_forces):.2f}G")
        print(f"   Max G-force: {max(self.g_forces):.2f}G")
        print(f"   Average stress: {np.mean(self.stress_scores):.1f}/100")
        print(f"   Max stress: {max(self.stress_scores):.1f}/100")
        print(f"   High stress events: {len(self.high_stress_events)}")
    
    def run_validation_tests(self):
        """Run validation tests"""
        
        print("\n📊 Running validation tests...")
        
        # Test 1: Data quality
        self.validation_tests.append({
            'name': 'Data Quality',
            'passed': len(self.records) > 100,
            'details': f'{len(self.records)} records loaded'
        })
        
        # Test 2: G-force range
        max_g = max(self.g_forces)
        avg_g = np.mean(self.g_forces)
        self.validation_tests.append({
            'name': 'G-Force Range',
            'passed': 0.5 < max_g < 20,
            'details': f'Avg: {avg_g:.2f}G, Max: {max_g:.2f}G'
        })
        
        # Test 3: Stress calculation
        avg_stress = np.mean(self.stress_scores)
        self.validation_tests.append({
            'name': 'Stress Calculation',
            'passed': 0 <= avg_stress <= 100,
            'details': f'Average stress: {avg_stress:.1f}/100'
        })
        
        # Test 4: Accelerometer data quality
        valid_accel = sum(1 for r in self.records if r.accel_x is not None)
        accel_percentage = (valid_accel / len(self.records)) * 100
        self.validation_tests.append({
            'name': 'Accelerometer Data',
            'passed': accel_percentage > 50,
            'details': f'{accel_percentage:.1f}% of records have IMU data'
        })
        
        # Test 5: High stress detection
        self.validation_tests.append({
            'name': 'High Stress Detection',
            'passed': True,
            'details': f'{len(self.high_stress_events)} high stress events detected'
        })
        
        # Print results
        passed = sum(1 for t in self.validation_tests if t['passed'])
        print(f"\n✅ {passed}/{len(self.validation_tests)} tests passed")
        
        for test in self.validation_tests:
            status = "✅" if test['passed'] else "❌"
            print(f"   {status} {test['name']}: {test['details']}")
    
    def generate_visualizations(self):
        """Generate validation charts"""
        
        print("\n📈 Generating visualizations...")
        
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Stress over time
        axes[0].plot(self.stress_scores, linewidth=0.5, color='red', alpha=0.7)
        axes[0].set_title('Overall Stress Score Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Stress Score (0-100)')
        axes[0].axhline(y=70, color='darkred', linestyle='--', linewidth=2, label='High Stress Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)
        
        # Plot 2: G-force over time
        axes[1].plot(self.g_forces, linewidth=0.5, color='blue', alpha=0.7)
        axes[1].set_title('G-Force Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('G-Force')
        axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='1G (Hover)', alpha=0.5)
        axes[1].axhline(y=2.0, color='orange', linestyle='--', linewidth=1, label='2G (Moderate)', alpha=0.5)
        axes[1].axhline(y=3.0, color='red', linestyle='--', linewidth=1, label='3G (High)', alpha=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Stress distribution histogram
        axes[2].hist(self.stress_scores, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[2].set_title('Stress Score Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Stress Score')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(x=np.mean(self.stress_scores), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(self.stress_scores):.1f}')
        axes[2].axvline(x=70, color='darkred', linestyle='--', linewidth=2, label='High Stress Threshold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f'validation_results_{Path(self.log_file).stem}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        
        # Save summary
        self.results = {
            'log_file': Path(self.log_file).name,
            'config': self.config.model_name,
            'total_records': len(self.records),
            'duration_seconds': (self.records[-1].timestamp - self.records[0].timestamp).total_seconds() if len(self.records) > 1 else 0,
            'avg_stress': float(np.mean(self.stress_scores)),
            'max_stress': float(max(self.stress_scores)),
            'min_stress': float(min(self.stress_scores)),
            'avg_g_force': float(np.mean(self.g_forces)),
            'max_g_force': float(max(self.g_forces)),
            'min_g_force': float(min(self.g_forces)),
            'high_stress_events': len(self.high_stress_events),
            'validation_tests': self.validation_tests,
            'high_stress_details': self.high_stress_events[:10]  # First 10 events
        }
        
        summary_file = output_dir / f'validation_summary_{Path(self.log_file).stem}.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Saved: {summary_file}")
    
    def run_validation(self):
        """Run complete validation workflow"""
        
        print("="*60)
        print("🔬 VALIDATION WITH PUBLIC DATA")
        print("="*60)
        
        self.load_config()
        self.load_data()
        self.calculate_stress()
        self.run_validation_tests()
        self.generate_visualizations()
        
        print("\n" + "="*60)
        print("✅ VALIDATION COMPLETE")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Records: {self.results['total_records']}")
        print(f"   Duration: {self.results['duration_seconds']:.1f} seconds")
        print(f"   Avg Stress: {self.results['avg_stress']:.1f}/100")
        print(f"   Max Stress: {self.results['max_stress']:.1f}/100")
        print(f"   Avg G-Force: {self.results['avg_g_force']:.2f}G")
        print(f"   Max G-Force: {self.results['max_g_force']:.2f}G")
        print(f"   High Stress Events: {self.results['high_stress_events']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n❌ Please provide log file and config file paths")
        print("\nUsage:")
        print("   python validate_with_public_data.py <log_file> <config_file>")
        print("\nExample:")
        print("   python validate_with_public_data.py \"..\\..\\data\\public\\kaggle_datasets\\SurveilDrone-Net23\\SurveilDrone-Net23.csv\" \"..\\..\\config\\drones\\quadcopter_450.json\"")
        sys.exit(1)
    
    log_file = sys.argv[1]
    config_file = sys.argv[2]
    
    validator = ValidationRunner(log_file, config_file)
    validator.run_validation()