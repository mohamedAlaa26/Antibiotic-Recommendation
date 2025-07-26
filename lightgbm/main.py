from data_preprocessing import prepare_data, encode_features
from train_models import train_lightgbm_ovr
from utils import save_models, load_models
from predict import predict_antibiotic
import numpy as np

def main():
    print("🧬 LightGBM One-vs-Rest Antibiotic Prediction with MLflow")
    
    df = prepare_data('cleaned_microbiology_cultures_data_mo.csv')
    processed_df, feature_columns, label_encoders = encode_features(df)

    models, scaler, performance_metrics = train_lightgbm_ovr(
        processed_df, feature_columns
    )

    save_models(models, scaler, label_encoders, feature_columns)

    print("\n📈 Performance Summary:")
    for antibiotic, metrics in performance_metrics.items():
        print(f"{antibiotic}: {metrics['accuracy']:.3f}")

    example_patient = {
        'median_heartrate': 95, 'median_resprate': 22, 'median_temp': 101.2,
        'median_sysbp': 130, 'median_diasbp': 85, 'gender_encoded': 1, 'age_encoded': 2,
        'median_wbc': 16.5, 'median_hgb': 11.0, 'median_plt': 200,
        'median_na': 138, 'median_hco3': 22, 'median_bun': 28, 'median_cr': 1.8
        
    }

    model_package = load_models()
    recommendations = predict_antibiotic(example_patient, model_package)

    print("\n🏆 Top 5 Antibiotic Recommendations:")
    for i, (antibiotic, prob) in enumerate(recommendations[:5], 1):
        print(f"{i}. {antibiotic} (Probability: {prob:.3f})")

if __name__ == "__main__":
    main()
