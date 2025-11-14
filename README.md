# Asset Preservation Digital Twin

A comprehensive digital twin system for drone asset preservation, featuring real-time telemetry processing, machine learning-based failure prediction, and multi-format flight log analysis.

## 🚁 Overview

This project provides:
- **Multi-format log parsing** for Betaflight, ArduPilot, PX4, UAV Navigation, and SurveilDrone datasets
- **Real-time telemetry streaming** via MQTT
- **Time-series data storage** with InfluxDB
- **ML-based failure detection** using pre-trained models
- **FastAPI backend** for data ingestion and analysis
- **Frontend dashboard** for visualization (in development)

## 🏗️ Architecture

```
├── backend/                 # Main application backend
│   ├── src/
│   │   ├── api/            # FastAPI routes
│   │   ├── data_ingestion/ # Log parsers (Betaflight, ArduPilot, etc.)
│   │   ├── database/       # Database initialization
│   │   ├── ml/             # ML models and feature engineering
│   │   ├── models/         # Data models
│   │   └── processing/     # Data processing pipelines
│   └── requirements.txt    # Python dependencies
├── frontend/               # React/Vue dashboard (TBD)
├── scripts/                # CLI tools for analysis & testing
├── data/                   # Data directories
│   ├── raw/               # Raw flight logs
│   ├── processed/         # Parsed/labeled data
│   ├── public/            # Public datasets (Kaggle, etc.)
│   └── failures/          # Failure case studies
├── config/                # Configuration files
│   └── drones/           # Drone-specific configs
└── docker-compose.yml    # Infrastructure setup
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Clone & Setup

```powershell
git clone <your-repo-url>
cd asset-preservation-digital-twin
```

### 2. Configure Environment

Copy the example environment file and customize it:

```powershell
cp .env.example .env
```

Edit `.env` with your settings (see Configuration section below).

### 3. Start Infrastructure

Launch databases, MQTT broker, and Redis:

```powershell
docker-compose up -d
```

Verify services are running:
```powershell
docker-compose ps
```

### 4. Install Python Dependencies

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5. Initialize Database

```powershell
python src/database/init_db.py
```

### 6. Run the API Server

```powershell
cd backend
uvicorn src.api.main:app --reload --port 8000
```

API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📋 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# InfluxDB (Time-series database)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=<generate-secure-token>
INFLUXDB_ORG=drone-org
INFLUXDB_BUCKET=telemetry

# PostgreSQL (Metadata database)
POSTGRES_USER=drone_user
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=drone_digital_twin
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# MQTT Broker
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC_TELEMETRY=drone/telemetry

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# Application
LOG_LEVEL=INFO
API_PORT=8000
```

### Docker Services

The `docker-compose.yml` provides:
- **InfluxDB** (port 8086): Time-series telemetry storage
- **PostgreSQL** (port 5432): Relational metadata storage
- **Mosquitto MQTT** (ports 1883, 9001): Real-time message broker
- **Redis** (port 6379): Caching layer

## 🛠️ Usage

### Parsing Flight Logs

Parse various drone log formats:

```powershell
# Betaflight logs
python scripts/inspect_betaflight.py data/public/betaflight_logs/your_log.BFL

# ArduPilot logs
python scripts/inspect_flight_log.py data/failures/case_001_ardupilot_crash/166.BIN

# Generic CSV parsing
python backend/src/data_ingestion/csv_parser.py data/raw/sample_flight_log.csv
```

### Feature Extraction & ML

```powershell
# Extract features from flight data
python scripts/test_feature_extraction.py

# Train failure detection model
python scripts/train_failure_model.py

# Detect anomalies in flight logs
python scripts/detect_anomalies.py
```

### Real-time Monitoring

```powershell
# Start real-time telemetry monitor
python scripts/run_realtime_monitor.py

# Simulate flight data streaming
python scripts/simulate_flight.py
```

### Data Validation

```powershell
# Validate with public datasets
python scripts/validate_with_public_data.py

# Analyze Kaggle datasets
python scripts/analyze_kaggle_datasets.py
```

## 🧪 Testing

Run the test suite:

```powershell
cd backend
pytest
```

Run with coverage:

```powershell
pytest --cov=src --cov-report=html
```

## 📊 Supported Flight Controllers

| Controller | Parser | Status | Log Format |
|------------|--------|--------|------------|
| Betaflight | ✅ | Stable | `.BFL`, `.BBL` |
| ArduPilot | ✅ | Stable | `.BIN`, `.bin` |
| PX4 | ✅ | Stable | `.ulg` |
| UAV Navigation | ✅ | Beta | Custom CSV |
| SurveilDrone | ✅ | Beta | Custom format |

## 🤖 Machine Learning

The system includes a pre-trained failure prediction model (`backend/src/ml/failure_model.joblib`) that:
- Extracts 50+ features from flight telemetry
- Detects anomalies in real-time
- Predicts component failures before they occur
- Classifies flight phases automatically

## 📁 Data Organization

```
data/
├── raw/              # Original flight logs (gitignored)
├── processed/        # Parsed CSV files (gitignored)
├── public/           # Public datasets for validation
│   ├── betaflight_logs/
│   └── kaggle_datasets/
├── failures/         # Failure case studies
│   ├── case_001_ardupilot_crash/
│   ├── case_002_motor_fail/
│   └── ...
└── validation/       # Validation reports and summaries
```

## 🔧 Common Tasks

### Create Drone Configuration

```powershell
python scripts/create_drone_configs.py
```

### Configure Custom Drone

```powershell
python scripts/configure_my_drone.py
```

### Classify Flight Phases

```powershell
python scripts/classify_flight_phases.py data/processed/parsed_my_flight_001.csv
```

### Label Flight Logs

```powershell
python scripts/label_flight_log.py data/raw/flight_log.csv
```

## 🐛 Troubleshooting

### Database Connection Issues

```powershell
# Check if services are running
docker-compose ps

# View logs
docker-compose logs influxdb
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Python Import Errors

Ensure you're in the correct virtual environment:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### MQTT Connection Failed

Check Mosquitto configuration:
```powershell
docker-compose logs mosquitto
```

Verify `config/mosquitto.conf` settings.

## 🚧 Development Status

### ✅ Completed
- Multi-format log parsing
- Database integration (InfluxDB, PostgreSQL)
- ML failure prediction model
- MQTT streaming
- Docker infrastructure

### 🔄 In Progress
- Frontend dashboard
- Real-time visualization
- API documentation
- Unit test coverage

### 📝 Planned
- Kubernetes deployment
- Cloud integration (AWS/Azure)
- Advanced ML models
- Mobile app integration

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Contact

[Add contact information here]

## 🙏 Acknowledgments

This project uses public datasets from:
- Kaggle UAV datasets
- Betaflight blackbox logs
- ArduPilot community logs
