"""
Test suite for CSV parser
"""
import pytest
import tempfile
import os
from pathlib import Path
from src.data_ingestion.csv_parser import CSVParser


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing"""
    csv_content = """timestamp,latitude,longitude,altitude,ground_speed,battery_voltage
2024-01-01 12:00:00,40.7128,-74.0060,100.5,5.2,11.8
2024-01-01 12:00:01,40.7129,-74.0061,101.2,5.5,11.7
2024-01-01 12:00:02,40.7130,-74.0062,102.0,5.8,11.6
2024-01-01 12:00:03,40.7131,-74.0063,102.5,6.0,11.5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_csv_parser_initialization(sample_csv_file):
    """Test CSVParser can be initialized with a file"""
    parser = CSVParser(sample_csv_file)
    assert parser is not None
    assert os.path.exists(sample_csv_file)


def test_csv_parser_parse(sample_csv_file):
    """Test parsing CSV returns records"""
    parser = CSVParser(sample_csv_file)
    records = parser.parse()
    
    assert records is not None
    assert len(records) == 4
    # Records are TelemetryRecord objects, check attributes exist
    assert hasattr(records[0], 'timestamp')
    assert hasattr(records[0], 'latitude')
    assert hasattr(records[0], 'longitude')


def test_csv_parser_to_dataframe(sample_csv_file):
    """Test conversion to pandas DataFrame"""
    parser = CSVParser(sample_csv_file)
    records = parser.parse()
    df = parser.to_dataframe()
    
    assert df is not None
    assert len(df) == 4
    assert 'latitude' in df.columns
    assert 'longitude' in df.columns
    assert 'altitude' in df.columns


def test_csv_parser_get_summary(sample_csv_file):
    """Test summary statistics generation"""
    parser = CSVParser(sample_csv_file)
    records = parser.parse()
    summary = parser.get_summary()
    
    assert summary is not None
    assert 'total_records' in summary or 'records' in summary.values()
    assert isinstance(summary, dict)


def test_csv_parser_with_nonexistent_file():
    """Test parser handles non-existent file gracefully"""
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        parser = CSVParser("nonexistent_file.csv")
        parser.parse()


def test_csv_parser_save_to_csv(sample_csv_file):
    """Test saving parsed data to CSV"""
    parser = CSVParser(sample_csv_file)
    records = parser.parse()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name
    
    try:
        parser.save_to_csv(output_path)
        assert os.path.exists(output_path)
        
        # Verify output file has data
        with open(output_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
