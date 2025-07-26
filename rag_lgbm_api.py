# rag_lgbm_api.py

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions

# === Load LightGBM + Ensemble models ===
model_package = joblib.load("models/lightgbm_ovr_models.pkl")
models = model_package['models']
label_encoders = model_package['label_encoders']
feature_columns = model_package['feature_columns']

# === Load ChromaDB and Embeddings ===
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("antibiotic_cases", embedding_function=embedding_fn)

# === Load Local LLaMA model ===
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=6
)


# === FastAPI Setup ===
app = FastAPI(title="🧠 Antibiotic Predictor + RAG Explainer")

# === Input Schema ===
class PatientInfo(BaseModel):
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

class ExplainRequest(BaseModel):
    patient: PatientInfo
    question: str

# === Utility Functions ===
def predict_antibiotics(patient_dict):
    patient_dict["age"] = patient_dict["age"].replace("years", "").strip()
    patient_dict["gender"] = patient_dict["gender"].strip().upper()

    # Label encoding
    patient_dict["age_encoded"] = label_encoders["age"].transform([patient_dict["age"]])[0]
    patient_dict["gender_encoded"] = label_encoders["gender"].transform([patient_dict["gender"]])[0]

    # DataFrame for prediction
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
    return ranked[:5]

def generate_explanation(question, retrieved_docs):
    prompt = f"""
You are a clinical assistant AI.
Base your answer ONLY on the retrieved patient cases below.
Explain clearly why a certain antibiotic may be recommended.

Retrieved Cases:
{retrieved_docs}

Question: {question}
Answer:
"""
    response = llm(prompt, max_tokens=512, temperature=0.2,stop=["\nQuestion:"])
    return response['choices'][0]['text'].strip()

# === Main Endpoint ===
@app.post("/explain_prediction")
def explain_prediction(payload: ExplainRequest):
    try:
        patient_data = payload.patient.dict()
        user_question = payload.question

        # Step 1: Run prediction
        top_predictions = predict_antibiotics(patient_data)

        # Step 2: Use top-1 antibiotic name to build retrieval query
        top_antibiotic = top_predictions[0][0]
        query_text = (
            f"{patient_data['age']} year old {patient_data['gender']} with vitals: "
            f"HR {patient_data['median_heartrate']}, Temp {patient_data['median_temp']}, "
            f"WBC {patient_data['median_wbc']} – treated with {top_antibiotic}"
        )

        # Step 3: Retrieve similar past cases from ChromaDB
        results = collection.query(query_texts=[query_text], n_results=3)
        retrieved_cases = "\n".join(results['documents'][0])

        # Step 4: Ask LLM to generate explanation
        explanation = generate_explanation(user_question, retrieved_cases)

        return {
            "status": "success",
            "top_5_recommendations": [
                {"antibiotic": ab, "probability": round(prob, 4)}
                for ab, prob in top_predictions
            ],
            "explanation": {
                "retrieved_cases": retrieved_cases,
                "conclusion": explanation
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction/Explanation error: {e}")
