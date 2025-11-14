from pathlib import Path

# Project root is 2 levels up from scripts/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / 'data'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PUBLIC_DATA_DIR = DATA_DIR / 'public'
VALIDATION_DATA_DIR = DATA_DIR / 'validation'

# Config subdirectories
DRONE_CONFIG_DIR = CONFIG_DIR / 'drones'

# Kaggle datasets
KAGGLE_DIR = PUBLIC_DATA_DIR / 'kaggle_datasets'