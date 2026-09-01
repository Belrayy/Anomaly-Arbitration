# Anomaly Arbitration

A comprehensive machine learning platform for detecting and analyzing anomalies across multiple datasets using ensemble methods. Anomaly Arbitration combines various detection algorithms to identify outliers and irregular patterns in credit card transactions, cybersecurity events, school data, and semiconductor measurements.

## 🎯 Features

- **Multi-Algorithm Detection**: Uses three detection algorithms:
  - Isolation Forest (IF)
  - Local Outlier Factor (LOF)
  - Support Vector Machine (SVM)
  
- **Multiple Datasets**: Supports analysis on:
  - Credit Card Transactions
  - Cybersecurity Events
  - School Data
  - Transistor/Semiconductor Data

- **Data Pipeline**: Complete data processing workflow:
  - Raw data ingestion
  - Data cleaning and validation
  - Cherry-picking for balanced datasets
  - Train/test splitting

- **REST API**: FastAPI backend with:
  - User authentication and security
  - Database persistence
  - Report generation
  - Prediction endpoints

- **Web Interface**: React/TypeScript frontend with:
  - Interactive dashboards
  - Real-time predictions
  - Data visualization
  - User authentication

- **Report Generation**: Automated report creation with detailed analysis and findings


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Docker & Docker Compose (optional)
- PostgreSQL (or compatible database)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Belrayy/Anomaly-Arbitration.git
   cd projet
   ```

2. **Backend Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Database Setup**
   ```bash
   # Run migrations
   cd api/database
   alembic upgrade head
   
   # Initialize database with SQL scripts
   psql -U postgres -d anomaly_db -f ../../post/create_users.sql
   psql -U postgres -d anomaly_db -f ../../post/create_reports.sql
   ```

### Running the Application

**Option 1: Using Docker Compose**
```bash
docker-compose up
```

**Option 2: Manual Start**

Start the backend:
```bash
cd api
uvicorn main:app --reload
```

Start the frontend (in a new terminal):
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📊 Usage

### Data Processing Pipeline

1. **Clean Data**
   ```bash
   python datasets/cleaning/credit_card_cleaning.py
   python datasets/cleaning/cyber_cleaning.py
   # ... etc for other datasets
   ```

2. **Cherry Pick Balanced Datasets**
   ```bash
   python datasets/cherry_picking/cherry_picking_credit_card.py
   ```

3. **Split Train/Test Data**
   ```bash
   python datasets/cleaning/test&train.py
   ```

### Model Training

Train models for each algorithm and dataset:
```bash
# Isolation Forest
python models/training/isolation_forest/<script>

# Local Outlier Factor
python models/training/local_outlier_factor/<script>

# Support Vector Machine
python models/training/svm/<script>
```

### Running Predictions

```bash
# Get predictions for a dataset
curl http://localhost:8000/api/predict \
  -H "Authorization: Bearer <token>" \
  -d '{"dataset": "credit_card", "algorithm": "isolation_forest"}'
```

### Generating Reports

```bash
# Generate reports for analyzed datasets
python reports/credit_card/report_gen_creditcard.py
python reports/cyber/report_gen_cyber.py
# ... etc for other datasets
```

## 🔐 Authentication

The API uses JWT-based authentication. Include your token in the Authorization header:
```
Authorization: Bearer <your_token>
```

See `api/auth/security.py` for implementation details.

## 📁 Project Structure

- **api/** - FastAPI backend application
  - `main.py` - Application entry point
  - `auth/` - Authentication and authorization
  - `database/` - Database models and migrations
  - `reports/` - Report generation endpoints

- **frontend/** - React TypeScript web interface
  - Built with Vite for fast development
  - TypeScript for type safety
  - Responsive UI components

- **datasets/** - Data processing and preparation
  - `cleaning/` - Data validation and cleaning scripts
  - `cherry_picking/` - Dataset balancing and selection
  - `data/` - Raw and processed data storage

- **models/** - Machine learning models
  - `training/` - Model training pipelines
  - `inference/` - Prediction and inference scripts

- **reports/** - Report generation
  - HTML/PDF report templates
  - Report generation scripts per dataset

- **Predictions/** - Generated prediction files (JSON format)

## 🛠️ Technologies

- **Backend**: Python, FastAPI, SQLAlchemy, Alembic
- **Frontend**: TypeScript, React, Vite, CSS
- **Database**: PostgreSQL
- **ML**: Scikit-learn (Isolation Forest, LOF, SVM)
- **Containerization**: Docker, Docker Compose

## 📈 Model Algorithms

- **Isolation Forest**: Tree-based ensemble method for anomaly detection
- **Local Outlier Factor (LOF)**: Density-based anomaly detection
- **Support Vector Machine (SVM)**: Classification-based anomaly detection

Each algorithm is trained on the provided datasets and generates predictions stored in the `Predictions/` directory.

## 📝 API Documentation

Once the backend is running, interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Development

### Adding New Datasets

1. Add raw data to `datasets/data/raw/`
2. Create cleaning script in `datasets/cleaning/`
3. Create cherry-picking script in `datasets/cherry_picking/`
4. Add model training/inference scripts

### Adding New Models

1. Create training script in `models/training/`
2. Create inference script in `models/inference/`
3. Update API routes to support new model
4. Add prediction storage in `Predictions/`

## 📋 Requirements

See `requirements.txt` for all Python dependencies.

Key packages:
- fastapi
- sqlalchemy
- scikit-learn
- pandas
- numpy
- pydantic

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

This project is part of an educational/professional initiative.

## 📧 Support

For questions or issues, please contact the development team or create an issue in the repository.

---

**Happy Anomaly Detecting!** 🎯
