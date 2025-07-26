import joblib
import os

def save_models(models, scaler, label_encoders, feature_columns, path='models'):
    os.makedirs(path, exist_ok=True)
    joblib.dump({
        'models': models,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }, os.path.join(path, 'lightgbm_ovr_models.pkl'))
    print(f"✅ Models saved to {path}/lightgbm_ovr_models.pkl")

def load_models(path='models/lightgbm_ovr_models.pkl'):
    return joblib.load(path)