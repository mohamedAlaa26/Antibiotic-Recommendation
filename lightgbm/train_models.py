import lightgbm as lgb
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_lightgbm_ovr(df, feature_columns, target_column='antibiotic'):
    print("\n🤖 Training Ensemble (LGBM, RF, LR, XGB) One-vs-Rest models with MLflow...")
    
    antibiotics = df[target_column].unique()
    models = {}
    performance_metrics = {}

    X = df[feature_columns].fillna(df[feature_columns].median())

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Antibiotic_Ensemble_OVR")

    scalers = {}

    for antibiotic in antibiotics:
        y = (df[target_column] == antibiotic).astype(int)
        if y.sum() < 10:
            print(f"⚠️ Skipping {antibiotic}: Not enough samples")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scalers[antibiotic] = scaler

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': 42,
            'verbose': -1
        }

        with mlflow.start_run(run_name=f"Ensemble_{antibiotic}"):
            # LightGBM
            lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
            lgb_valid = lgb.Dataset(X_test_scaled, label=y_test)
            lgb_model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_valid],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100)]
            )

            # RandomForest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)

            # Logistic Regression
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)

            # XGBoost
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train_scaled, y_train)

            # Ensemble prediction (average)
            lgb_pred = lgb_model.predict(X_test_scaled)
            rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
            lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
            xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
            avg_pred = np.mean([lgb_pred, rf_pred, lr_pred, xgb_pred], axis=0)
            y_pred_binary = (avg_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred_binary)
            report = classification_report(y_test, y_pred_binary)

            print(f"✅ {antibiotic}: Accuracy={accuracy:.3f}")

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(report, f"classification_report_{antibiotic}.txt")
            # Optionally log models

            models[antibiotic] = {
                'lgb': lgb_model,
                'rf': rf_model,
                'lr': lr_model,
                'xgb': xgb_model,
                'scaler': scaler
            }
            performance_metrics[antibiotic] = {'accuracy': accuracy, 'report': report}

    # Calculate and print average accuracy
    if performance_metrics:
        avg_accuracy = np.mean([m['accuracy'] for m in performance_metrics.values()])
        print(f"\n🔎 Average accuracy across all antibiotics: {avg_accuracy:.3f}")
    else:
        avg_accuracy = None

    return models, scalers, performance_metrics