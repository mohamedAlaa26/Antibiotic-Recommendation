import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model package
MODEL_PATH = "models/lightgbm_ovr_models.pkl"
model_package = joblib.load(MODEL_PATH)
models = model_package['models']
feature_columns = model_package['feature_columns']
label_encoders = model_package['label_encoders']

# FastAPI app
app = FastAPI(title="Antibiotic Ensemble Prediction API")

class PatientData(BaseModel):
    age: str
    gender: str
    median_heartrate: float
    median_resprate: float
    median_temp: float
    median_sysbp: float
    median_diasbp: float
    median_wbc: float
    median_hgb: float
    median_plt: float
    median_na: float
    median_hco3: float
    median_bun: float
    median_cr: float

@app.post("/predict")
def predict(patient: PatientData):
    try:
        patient_dict = patient.dict()
        patient_dict["age"] = patient_dict["age"].replace("years", "").strip()
        patient_dict["gender"] = patient_dict["gender"].strip().upper()
        patient_dict["age_encoded"] = label_encoders["age"].transform([patient_dict["age"]])[0]
        patient_dict["gender_encoded"] = label_encoders["gender"].transform([patient_dict["gender"]])[0]

        patient_df = pd.DataFrame([{
            'median_heartrate': patient_dict['median_heartrate'],
            'median_resprate': patient_dict['median_resprate'],
            'median_temp': patient_dict['median_temp'],
            'median_sysbp': patient_dict['median_sysbp'],
            'median_diasbp': patient_dict['median_diasbp'],
            'median_wbc': patient_dict['median_wbc'],
            'median_hgb': patient_dict['median_hgb'],
            'median_plt': patient_dict['median_plt'],
            'median_na': patient_dict['median_na'],
            'median_hco3': patient_dict['median_hco3'],
            'median_bun': patient_dict['median_bun'],
            'median_cr': patient_dict['median_cr'],
            'age_encoded': patient_dict['age_encoded'],
            'gender_encoded': patient_dict['gender_encoded']
        }])

        antibiotics = []
        probs = []
        for antibiotic, model_info in models.items():
            scaler = model_info['scaler']
            patient_scaled = scaler.transform(patient_df)
            lgb_prob = model_info['lgb'].predict(patient_scaled)[0]
            rf_prob = model_info['rf'].predict_proba(patient_scaled)[0][1]
            lr_prob = model_info['lr'].predict_proba(patient_scaled)[0][1]
            xgb_prob = model_info['xgb'].predict_proba(patient_scaled)[0][1]
            avg_prob = np.mean([lgb_prob, rf_prob, lr_prob, xgb_prob])
            antibiotics.append(antibiotic)
            probs.append(avg_prob)

        ranked = sorted(zip(antibiotics, probs), key=lambda x: x[1], reverse=True)

        return {
            "status": "success",
            "top_5_recommendations": [
                {"antibiotic": antibiotic, "probability": round(prob, 4)}
                for antibiotic, prob in ranked[:5]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")