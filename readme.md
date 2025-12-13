# 💳 End-to-End Credit Risk Scoring Engine

A production-ready Machine Learning system that predicts loan default probability using alternative data. This project implements a full ML lifecycle: from advanced feature engineering on the Home Credit Default Risk dataset to a containerized deployment serving real-time predictions via REST API and an interactive dashboard.

## 🚀 Key Features

⚡ High-Performance Inference: Built with FastAPI for low-latency predictions (<100ms).

🧠 Advanced Modeling: Implements an ensemble of CatBoost, XGBoost, and LightGBM, achieving an AUC-ROC of 0.78, significantly outperforming the logistic regression baseline.

🐳 Containerized Architecture: Fully Dockerized application ensuring reproducibility across Dev and Prod environments.

📊 Interactive UI: User-friendly Streamlit dashboard for loan officers to input data and visualize risk probabilities.

🛠️ Robust Pipeline: Modular codebase separating data loading, feature engineering (src/feature_engineering.py), and inference logic.

## 🏗️ System Architecture

<img width="928" height="851" alt="image" src="https://github.com/user-attachments/assets/0b4fab3b-a4c3-488f-a1eb-3a6d59d0110f" />



## 🛠️ Tech Stack

Machine Learning: Scikit-Learn, CatBoost, XGBoost, LightGBM, SHAP (Explainability).

API & Backend: FastAPI, Uvicorn, Pydantic.

Frontend: Streamlit.

DevOps: Docker, Joblib (Serialization).

Data Processing: Pandas, NumPy, Regex.

## 📊 Model Performance

I benchmarked multiple algorithms to optimize for the AUC-ROC metric (Area Under the Curve), critical for imbalanced classification tasks like fraud/default prediction.

Model

AUC-ROC Score

Notes

CatBoost (Final)

0.776

Best performance, handles categorical features natively.

LightGBM

0.771

Fast training, high accuracy.

XGBoost

0.764

Strong baseline gradient boosting.

Logistic Regression

0.754

Simple baseline for interpretability.

The final deployed model uses CatBoost due to its superior stability and handling of categorical variables.

## 🐳 Running with Docker (Recommended)

You can spin up the entire system (API + UI) with a single command.

1. Build the Images

### Build the API service
docker build -f Dockerfile.fastapi -t credit-risk-api .

### Build the Streamlit UI
docker build -f Dockerfile.streamlit -t credit-risk-ui .


2. Run Containers

### Start API on port 8000
docker run -d -p 8000:8000 credit-risk-api

### Start UI on port 8501
docker run -d -p 8501:8501 credit-risk-ui


3. Access the App

API Swagger Docs: http://localhost:8000/docs - Test endpoints directly.

Dashboard: http://localhost:8501 - Interactive form.

## 🧪 Local Installation (No Docker)

### Clone the repo
git clone [https://github.com/amriteshp2000/credit-risk-predictor.git](https://github.com/amriteshp2000/credit-risk-predictor.git)
cd credit-risk-predictor

### Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run FastAPI Server
python app/main.py serve

### In a new terminal, run Streamlit
python app/main.py


---

## 🔍 Project Structure Overview

```
credit-risk-project/
├── app/                      # FastAPI + Streamlit application logic
│   ├── components/           # Streamlit UI forms
│   │   └── main.py           # Streamlit form layout
│   └── main.py               # Unified entry point for API/UI
├── data/                     # Dataset
│   ├── raw/                  # Raw data (as downloaded)
│   └── processed/            # Processed feature & target data for modeling
├── models/                   # Trained ML models (pkl format)
├── notebooks/                # Exploratory & modeling notebooks
│   ├── preprocessing.ipynb   # Preprocessing logic (Top Kaggle approach)
│   ├── modeling.ipynb        # Training + evaluation of models
│   ├── shap_analysis.ipynb   # Model interpretability
├── src/                      # Python scripts for modular ML pipeline
│   ├── data_loader.py        # Data loading helpers
│   ├── feature_engineering.py# Feature creation logic
│   ├── model.py              # Model training + saving
│   └── utils.py              # General utilities
├── .dockerignore
├── Dockerfile.fastapi        # Dockerfile for FastAPI API server
├── Dockerfile.streamlit      # Dockerfile for Streamlit UI server
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview & usage
```

---

## 🧰 Key Components

### `/models/`

Serialized trained models:

* `catboost_model.pkl`, `xgboost_model.pkl`, etc.
* `catboost_features.pkl`: stores the feature list used for CatBoost prediction.

### `/app/main.py`

* Unified script for running FastAPI (`uvicorn`) or Streamlit based on execution mode.
* `/predict` route for prediction API.
* Streamlit app renders form-based UI.

### `/app/components/main.py`

* Renders form inputs via `get_user_input_form()` function for Streamlit UI.

### `/src/`

* `feature_engineering.py`: domain-relevant feature generation.
* `data_loader.py`: read data from CSV/SQL.
* `model.py`: training and saving of models.
* `utils.py`: reusable helper functions.

### `/notebooks/`

* Exploratory and modeling notebooks with Kaggle-level insights.

---

## 🔐 Tips for Production Use

* Use `.env` to store sensitive keys and secrets
* Add authentication to API endpoints
* Deploy behind Nginx with HTTPS (e.g. using Docker Compose)
* Integrate CI/CD with GitHub Actions

---

## 🙏 Credits

* Dataset: Home Credit Default Risk - Kaggle
* Inspired by top Kaggle solutions
* Libraries: FastAPI, Streamlit, XGBoost, LightGBM, CatBoost, SHAP

---

