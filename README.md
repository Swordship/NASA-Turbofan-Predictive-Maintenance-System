# 🛩️ Turbofan Engine Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

A complete ML-powered predictive maintenance solution for industrial machinery monitoring using NASA's Turbofan Engine Dataset. Predicts Remaining Useful Life (RUL), detects anomalies, and triggers real-time alerts to prevent equipment failures.

![Dashboard Preview](assets/dashboard-preview.png)

## 🎯 Features

- 🤖 **Deep Learning RUL Prediction** - LSTM neural network with 93% accuracy
- 📊 **Real-time Monitoring** - 8 critical sensor parameters tracked continuously
- 🚨 **Intelligent Alerting** - Multi-level threshold violation detection
- 📈 **Interactive Dashboard** - Live charts and real-time visualizations
- 🎲 **6 Test Scenarios** - From normal operation to emergency conditions
- 🐳 **Docker Ready** - One-command deployment
- 🔌 **RESTful API** - Easy integration with existing systems

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance

# Run with Docker Compose
docker-compose up

# Access dashboard
open http://localhost:8080
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NASA dataset
# Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
# Place train_FD001.txt in data/ folder

# Run training pipeline
python src/1_data_preparation.py
python src/2_train_model.py
python src/3_define_thresholds.py
python src/4_mock_data_generator.py

# Start backend server
python src/5_backend_api.py

# Open dashboard
open frontend/dashboard.html
```

## 📋 Table of Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend Dashboard                     │
│              (React/HTML + Chart.js)                     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────┐
│                   Flask API Server                       │
│         (REST endpoints + Real-time streaming)          │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼──────┐ ┌──▼────────┐
│  ML Models   │ │  Mock   │ │ Threshold │
│ (LSTM + RF)  │ │  Data   │ │  Monitor  │
└──────────────┘ └─────────┘ └───────────┘
```

### Tech Stack

**Backend:**
- Python 3.8+
- TensorFlow/Keras (LSTM)
- Scikit-learn (Random Forest)
- Flask (REST API)
- Pandas, NumPy

**Frontend:**
- HTML5, CSS3, JavaScript
- Chart.js (Visualizations)
- Responsive Design

**Infrastructure:**
- Docker & Docker Compose
- PostgreSQL (Optional)
- Redis (Optional for caching)

## 📊 Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Dataset**.

### About the Dataset

- **Source:** NASA Prognostics Center of Excellence
- **Description:** Run-to-failure simulations of turbofan engines
- **Engines:** 100 units with varying lifespans
- **Sensors:** 21 sensor measurements per cycle
- **Total Cycles:** ~20,000 operational cycles

### Key Sensors Monitored

| Sensor | Description | Unit | Critical for |
|--------|-------------|------|--------------|
| T30 | HPC Outlet Temperature | °R | Compressor health |
| T50 | LPT Outlet Temperature | °R | Turbine degradation |
| P30 | HPC Outlet Pressure | psia | Seal wear |
| P2 | Fan Inlet Pressure | psia | Intake issues |
| Nf | Fan Speed | rpm | Bearing health |
| Nc | Core Speed | rpm | Rotor balance |

### Download Dataset

1. **Kaggle:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. **NASA:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
3. Place `train_FD001.txt` in `data/` folder

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- (Optional) Docker & Docker Compose

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create necessary directories
mkdir -p data models

# 6. Download dataset (see Dataset section)
# Place train_FD001.txt in data/ folder

# 7. Run training pipeline
python src/1_data_preparation.py      # Process raw data
python src/2_train_model.py           # Train ML models
python src/3_define_thresholds.py     # Define safety limits
python src/4_mock_data_generator.py   # Generate test scenarios

# 8. Start API server
python src/5_backend_api.py

# 9. Open dashboard in browser
open frontend/dashboard.html
```

## 🎮 Usage

### Starting the System

```bash
# Terminal 1: Start backend API
python src/5_backend_api.py

# Terminal 2: Serve frontend (optional)
cd frontend
python -m http.server 8080

# Access dashboard at http://localhost:8080/dashboard.html
```

### Using the Dashboard

1. **Select Scenario:** Choose from 6 test scenarios
2. **Start Monitoring:** Click "▶ Start Monitoring"
3. **Observe Predictions:** Watch RUL decrease in real-time
4. **Check Alerts:** Monitor threshold violations
5. **Analyze Trends:** View sensor charts and patterns

### Test Scenarios

| Scenario | Description | Expected RUL | Alerts |
|----------|-------------|--------------|--------|
| Normal | Healthy operation | 150 cycles | None |
| Gradual Degradation | Slow temperature rise | 80 cycles | Yellow warnings |
| Critical | Near failure thresholds | 30 cycles | Multiple warnings |
| Emergency | Imminent failure | 10 cycles | Critical alerts |
| Sudden Anomaly | FOD event spike | 100 cycles | Spike at cycle 50 |
| Pressure Drop | Seal failure | 50 cycles | Pressure warnings |

## 📡 API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### Get Available Scenarios

```http
GET /api/scenarios
```

**Response:**
```json
{
  "scenarios": [
    {
      "id": "normal",
      "name": "Normal Operation",
      "description": "All parameters within safe ranges",
      "expected_rul": 150
    }
  ]
}
```

#### Get Current Reading

```http
GET /api/current
```

**Response:**
```json
{
  "cycle": 42,
  "scenario": "normal",
  "reading": {
    "T2": 518.67,
    "T30": 1580.5,
    "T50": 1398.2,
    "P2": 14.62,
    "P30": 45.2
  },
  "prediction": {
    "rul": 108,
    "confidence": 0.87,
    "health_score": 72
  },
  "alerts": []
}
```

#### Start Simulation

```http
POST /api/control/start
Content-Type: application/json

{
  "scenario": "gradual_degradation"
}
```

#### Predict RUL

```http
POST /api/predict
Content-Type: application/json

{
  "T2": 518.67,
  "T24": 642.15,
  "T30": 1580.5,
  "T50": 1398.2,
  "P2": 14.62,
  "P30": 45.2,
  "Nf": 2388.06,
  "Nc": 9046.19
}
```

### Full API Reference

See [API.md](docs/API.md) for complete documentation.

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build backend image
docker build -t predictive-maintenance-backend -f docker/Dockerfile.backend .

# Build frontend image
docker build -t predictive-maintenance-frontend -f docker/Dockerfile.frontend .

# Run backend
docker run -p 5000:5000 predictive-maintenance-backend

# Run frontend
docker run -p 8080:80 predictive-maintenance-frontend
```

### Docker Architecture

```
┌─────────────────────────────────────┐
│         Nginx (Port 8080)           │
│      (Frontend + Reverse Proxy)     │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│      Flask API (Port 5000)          │
│     (Backend + ML Models)           │
└─────────────────────────────────────┘
```

## ☁️ Cloud Deployment

### Deploy to Heroku

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create turbofan-maintenance

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Deploy to AWS (EC2)

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. Clone repository
git clone https://github.com/YOUR_USERNAME/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance

# 5. Run with Docker Compose
sudo docker-compose up -d

# 6. Configure security group to allow ports 8080 and 5000
```

### Deploy to Google Cloud Run

```bash
# Build and deploy
gcloud run deploy turbofan-maintenance \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Deploy to Azure Container Instances

```bash
# Build image
docker build -t turbofan-maintenance .

# Push to Azure Container Registry
az acr build --registry myregistry --image turbofan-maintenance .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name turbofan-maintenance \
  --image myregistry.azurecr.io/turbofan-maintenance \
  --dns-name-label turbofan-maintenance \
  --ports 5000 8080
```

## 📈 Model Performance

### LSTM Neural Network

```
Architecture:
  Input: (30 cycles × 20 features)
  LSTM(128) → Dropout(0.2)
  LSTM(64) → Dropout(0.2)
  Dense(32) → Dense(16) → Dense(1)

Performance:
  ✅ MAE: 8.71 cycles
  ✅ RMSE: 17.45 cycles
  ✅ R² Score: 0.93
  ✅ Training Time: ~5 minutes (CPU)
```

### Random Forest Baseline

```
Configuration:
  - 100 estimators
  - Max depth: Auto
  - Random state: 42

Performance:
  MAE: ~12 cycles
  RMSE: ~21 cycles
  R² Score: 0.88
```

### Performance Comparison

| Metric | LSTM | Random Forest | Improvement |
|--------|------|---------------|-------------|
| MAE | 8.71 | 12.05 | 27.7% |
| RMSE | 17.45 | 21.33 | 18.2% |
| R² | 0.93 | 0.88 | 5.7% |

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Test API endpoints
pytest tests/test_api.py

# Test ML models
pytest tests/test_models.py
```

## 📁 Project Structure

```
turbofan-predictive-maintenance/
├── data/
│   ├── train_FD001.txt         # NASA dataset
│   ├── processed_data.csv      # Processed data
│   └── mock_*.csv              # Test scenarios
├── models/
│   ├── lstm_rul_model.h5       # Trained LSTM
│   ├── rf_rul_model.pkl        # Random Forest
│   ├── scaler.pkl              # Feature scaler
│   └── thresholds.json         # Safety thresholds
├── src/
│   ├── 1_data_preparation.py
│   ├── 2_train_model.py
│   ├── 3_define_thresholds.py
│   ├── 4_mock_data_generator.py
│   └── 5_backend_api.py
├── frontend/
│   └── dashboard.html
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── tests/
│   ├── test_api.py
│   ├── test_models.py
│   └── test_utils.py
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── ARCHITECTURE.md
├── assets/
│   └── dashboard-preview.png
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/

# Run tests
pytest tests/ --cov=src
```

## 🐛 Known Issues

- Random Forest training can be slow on CPU (use LITE version)
- Dashboard requires modern browser (Chrome/Firefox recommended)
- Large datasets may require GPU for training

## 🗺️ Roadmap

- [ ] Add SMS/Email notification system
- [ ] Multi-engine fleet monitoring
- [ ] Export reports to PDF
- [ ] Integration with maintenance management systems
- [ ] Mobile app (React Native)
- [ ] Real hardware sensor integration
- [ ] Advanced anomaly explanation (SHAP values)
- [ ] Cost-benefit analysis calculator

## 📚 References

- [NASA C-MAPSS Dataset Paper](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [LSTM for Time Series](https://arxiv.org/abs/1909.09586)
- [Predictive Maintenance Best Practices](https://www.sciencedirect.com/science/article/pii/S2212827120300378)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- NASA Ames Prognostics Center of Excellence for the dataset
- TensorFlow team for the excellent deep learning framework
- The open-source community for invaluable tools and libraries

## 📧 Contact

- **Email:** your.email@example.com
- **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Link:** https://github.com/YOUR_USERNAME/turbofan-predictive-maintenance

---

⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ for industrial IoT and predictive maintenance**