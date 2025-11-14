"""
Test suite for configuration management
"""
import pytest
import os
from src.config import Settings


def test_settings_loads_from_env():
    """Test that settings loads from environment variables"""
    # This assumes .env file exists
    settings = Settings()
    
    assert settings.INFLUXDB_ORG == "drone-org"
    assert settings.INFLUXDB_BUCKET == "telemetry"
    assert settings.POSTGRES_DB == "drone_digital_twin"
    assert settings.ENVIRONMENT == "development"


def test_required_fields_must_be_set():
    """Test that required fields (token, password) cannot be empty"""
    settings = Settings()
    
    # These should be set from .env
    assert settings.INFLUXDB_TOKEN
    assert len(settings.INFLUXDB_TOKEN) > 0
    assert settings.POSTGRES_PASSWORD
    assert len(settings.POSTGRES_PASSWORD) > 0


def test_optional_fields_can_be_none():
    """Test that optional fields default to None"""
    settings = Settings()
    
    # These are optional and may be None
    assert settings.MQTT_USERNAME is None or isinstance(settings.MQTT_USERNAME, str)
    assert settings.MQTT_PASSWORD is None or isinstance(settings.MQTT_PASSWORD, str)
    assert settings.REDIS_PASSWORD is None or isinstance(settings.REDIS_PASSWORD, str)


def test_default_values():
    """Test default configuration values"""
    settings = Settings()
    
    assert settings.INFLUXDB_URL == "http://localhost:8086"
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.POSTGRES_PORT == 5432
    assert settings.MQTT_PORT == 1883
    assert settings.REDIS_PORT == 6379
    assert settings.LOG_LEVEL == "INFO"
    assert settings.API_PORT == 8000


def test_postgres_connection_string():
    """Test that we can construct a valid PostgreSQL connection string"""
    settings = Settings()
    
    conn_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    
    assert "postgresql://" in conn_string
    assert settings.POSTGRES_USER in conn_string
    assert settings.POSTGRES_HOST in conn_string
    assert str(settings.POSTGRES_PORT) in conn_string
    assert settings.POSTGRES_DB in conn_string
