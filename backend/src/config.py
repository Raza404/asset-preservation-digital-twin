from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Create a .env file based on .env.example to configure these values.
    """
    
    # Database - InfluxDB (Time-series)
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str  # REQUIRED: Must be set in .env
    INFLUXDB_ORG: str = "drone-org"
    INFLUXDB_BUCKET: str = "telemetry"
    
    # Database - PostgreSQL (Relational)
    POSTGRES_USER: str = "drone_user"
    POSTGRES_PASSWORD: str  # REQUIRED: Must be set in .env
    POSTGRES_DB: str = "drone_digital_twin"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # MQTT Broker
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC_TELEMETRY: str = "drone/telemetry"
    MQTT_USERNAME: Optional[str] = None
    MQTT_PASSWORD: Optional[str] = None
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # Application Settings
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    API_HOST: str = "127.0.0.1"
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
