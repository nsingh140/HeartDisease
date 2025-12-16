from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk",
    version="1.0.0"
)

# Input schema
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Load model at startup
try:
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully from local file")
    else:
        # Fallback to MLflow
        import mlflow.pyfunc
        model = mlflow.pyfunc.load_model("models:/HeartDiseaseClassifier_RandomForest/latest")
        logger.info("Model loaded from MLflow")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "status": "healthy"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Log request
        logger.info(f"Prediction request: {data.dict()}")
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_data)
            confidence = float(np.max(prediction_proba))
        except:
            confidence = None
        
        result = {
            "prediction": int(prediction[0]),
            "risk_level": "High" if prediction[0] == 1 else "Low",
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    return {
        "model_type": str(type(model)),
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "target": "Heart disease (0=No, 1=Yes)"
    }