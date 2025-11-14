"""
Test configuration for pytest
"""
import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set test environment variables if not already set
if not os.getenv('INFLUXDB_TOKEN'):
    os.environ['INFLUXDB_TOKEN'] = 'test-token-for-ci'
if not os.getenv('POSTGRES_PASSWORD'):
    os.environ['POSTGRES_PASSWORD'] = 'test-password-for-ci'
