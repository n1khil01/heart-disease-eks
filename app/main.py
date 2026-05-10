from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

app = FastAPI(title="Heart Disease Risk Predictor")

BUNDLE_PATH = Path("model.pkl")
bundle = None

def load_bundle():
    global bundle
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

load_bundle()

class PatientInput(BaseModel):
    age: float
    trestbps: float
    chol: Optional[float] = None
    thalch: float
    oldpeak: float
    ca: Optional[float] = None
    sex: str
    cp: str
    fbs: str
    restecg: str
    exang: str
    slope: str
    thal: str

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

def preprocess(data: PatientInput, preprocessor) -> pd.DataFrame:
    row = {
        "age": data.age,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "thalch": data.thalch,
        "oldpeak": data.oldpeak,
        "ca": data.ca,
        "sex": data.sex,
        "cp": data.cp,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "exang": data.exang,
        "slope": data.slope,
        "thal": data.thal,
    }
    X = pd.DataFrame([row])

    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dummy_na=True, dtype=float)

    X[NUMERIC_FEATURES] = preprocessor.imputer.transform(X[NUMERIC_FEATURES])
    X = X.fillna(0)
    X[NUMERIC_FEATURES] = preprocessor.scaler.transform(X[NUMERIC_FEATURES])

    X = X.reindex(columns=preprocessor.feature_columns, fill_value=0)
    return X

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: PatientInput):
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    X = preprocess(patient, preprocessor)
    prob = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])
    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "risk": "high" if prediction == 1 else "low"
    }