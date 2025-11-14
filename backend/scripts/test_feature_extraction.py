import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.betaflight_parser import BetaflightParser
from src.ml.feature_engineering import UniversalFeatureExtractor, FeatureConfig

def test_feature_extraction(log_file: str):
    """Test feature extraction on flight log"""
    
    print("="*60)
    print("🧪 TESTING UNIVERSAL FEATURE EXTRACTOR")
    print("="*60)
    
    # Parse log
    print("\n1️⃣ Parsing flight log...")
    parser = BetaflightParser(log_file)
    records = parser.parse()
    print(f"   ✅ Loaded {len(records)} records")
    
    # Extract features
    print("\n2️⃣ Extracting features...")
    config = FeatureConfig(
        window_size_seconds=1.0,
        rolling_window_size=10,
        include_statistical=True,
        include_derivative=True
    )
    
    extractor = UniversalFeatureExtractor(config)
    features = extractor.extract_features(records)
    
    print(f"\n3️⃣ Feature Extraction Results:")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Total samples: {len(features)}")
    print(f"   Memory usage: {features.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    print(f"\n4️⃣ Feature Categories:")
    feature_groups = {}
    for col in features.columns:
        prefix = col.split('_')[0]
        feature_groups[prefix] = feature_groups.get(prefix, 0) + 1
    
    for group, count in sorted(feature_groups.items()):
        print(f"   {group:20s}: {count:3d} features")
    
    print(f"\n5️⃣ Sample Features (first 5 rows):")
    print(features.head())
    
    print(f"\n6️⃣ Feature Statistics:")
    print(features.describe().T)
    
    print("\n✅ Feature extraction test complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_feature_extraction.py <log_file>")
        print("\nExample:")
        print("   python test_feature_extraction.py \"..\\..\\data\\public\\betaflight_logs\\my_flight_001.csv\"")
        sys.exit(1)
    
    test_feature_extraction(sys.argv[1])