from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class DroneMetadata(Base):
    __tablename__ = "drone_metadata"
    
    id = Column(Integer, primary_key=True)
    drone_id = Column(String, unique=True, nullable=False)
    model = Column(String)
    frame_material = Column(String)
    weight_kg = Column(Float)
    battery_capacity_mah = Column(Integer)
    motor_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ComponentHealth(Base):
    __tablename__ = "component_health"
    
    id = Column(Integer, primary_key=True)
    drone_id = Column(String, nullable=False)
    component_type = Column(String, nullable=False)
    component_id = Column(String)
    health_score = Column(Float)
    total_runtime_hours = Column(Float)
    last_maintenance = Column(DateTime)
    predicted_failure_date = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FlightSession(Base):
    __tablename__ = "flight_sessions"
    
    id = Column(Integer, primary_key=True)
    drone_id = Column(String, nullable=False)
    session_id = Column(String, unique=True, nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    distance_meters = Column(Float)
    max_altitude_meters = Column(Float)
    total_stress_score = Column(Float)
    high_g_events = Column(Integer)
    battery_consumed_mah = Column(Float)
    notes = Column(String)

class MaintenanceLog(Base):
    __tablename__ = "maintenance_log"
    
    id = Column(Integer, primary_key=True)
    drone_id = Column(String, nullable=False)
    component = Column(String, nullable=False)
    action = Column(String)
    description = Column(String)
    performed_at = Column(DateTime, default=datetime.utcnow)
    performed_by = Column(String)

class DroneConfiguration(Base):
    __tablename__ = "drone_configurations"
    
    id = Column(Integer, primary_key=True)
    drone_id = Column(String, unique=True, nullable=False)
    model_name = Column(String)
    frame_type = Column(String)
    num_motors = Column(Integer)
    wheelbase_m = Column(Float)
    total_weight_kg = Column(Float)
    config_json = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
