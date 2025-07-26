import pandas as pd
import numpy as np

def predict_antibiotic(patient_data, model_package):
    models = model_package['models']
    feature_columns = model_package['feature_columns']

    patient_features = pd.DataFrame([patient_data])[feature_columns].fillna(0)

    predictions = {}
    for antibiotic, model_info in models.items():
        scaler = model_info['scaler']
        patient_scaled = scaler.transform(patient_features)

        # Get probabilities from each model
        lgb_prob = model_info['lgb'].predict(patient_scaled)[0]
        rf_prob = model_info['rf'].predict_proba(patient_scaled)[0][1]
        lr_prob = model_info['lr'].predict_proba(patient_scaled)[0][1]
        xgb_prob = model_info['xgb'].predict_proba(patient_scaled)[0][1]

        # Average (soft voting)
        avg_prob = np.mean([lgb_prob, rf_prob, lr_prob, xgb_prob])
        predictions[antibiotic] = avg_prob

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return ranked