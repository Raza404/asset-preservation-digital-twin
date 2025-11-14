import sys
import os
from pathlib import Path

file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/public/kaggle_datasets/SurveilDrone-Net23/SurveilDrone-Net23.csv"

print("Current directory:", os.getcwd())
print("Input path:", file_path)
print("Absolute path:", os.path.abspath(file_path))
print("Exists?", os.path.exists(file_path))

# Show what's actually in the directory
data_dir = Path("..") / "data" / "public" / "kaggle_datasets" / "SurveilDrone-Net23"
if data_dir.exists():
    print("\nFiles in directory:")
    for f in data_dir.iterdir():
        print(f"  - {f.name}")
else:
    print(f"\nDirectory doesn't exist: {data_dir}")