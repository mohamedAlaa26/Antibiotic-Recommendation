# 🧠 Antibiotic Recommendation and RAG Explanation System

This project combines ensemble machine learning and Retrieval-Augmented Generation (RAG) to predict the most suitable antibiotic for a patient based on clinical features and provide a natural language explanation powered by a local LLaMA-based language model.

---

## 🔍 Project Overview

- **Antibiotic Prediction**: Predicts the top 5 recommended antibiotics using an ensemble of 4 models:
  - LightGBM
  - XGBoost
  - Logistic Regression
  - Random Forest

- **RAG Explanation**: Retrieves similar past clinical cases using [ChromaDB](https://www.trychroma.com/) and generates an explanation for the recommendation using a local LLaMA model (`mistral-7b-instruct-v0.1`).

---


---

## 📌 Important Notes

- ⚠️ **Model file excluded**:  
  The file `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf` is **not included in this repository** due to its large size.  
  To run the RAG explanation system locally, you'll need to download this model manually and place it in the `models/` directory.

- 🧠 **Training logic**:  
  The training logic for the ensemble models (LightGBM, XGBoost, Logistic Regression, and Random Forest) is implemented in `train_lightgbm_ovr()` inside the training script.  
  It follows a **One-vs-Rest (OVR)** approach for each antibiotic class and uses MLflow for tracking.

---

## 🚀 How to Run the API

1. Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
uvicorn rag_lgbm_api:app --reload
```

3. Visit http://127.0.0.1:8000/docs to test the API using Swagger UI.


## 🧪 Example Input

{
  "patient": {
    "median_heartrate": 89.0,
    "median_resprate": 19.0,
    "median_temp": 102.2,
    "median_sysbp": 124.0,
    "median_diasbp": 74.0,
    "median_wbc": 17.2,
    "median_hgb": 14500.0,
    "median_plt": 222.0,
    "median_na": 132.0,
    "median_hco3": 21.0,
    "median_bun": 28.0,
    "median_cr": 1.47
  },
  "question": "Is Ertapenem effective in cases like this?"
}


## 📚 Technologies Used

Python, Pandas, Scikit-learn

LightGBM, XGBoost

FastAPI

ChromaDB (Vector DB)

LLaMA / Mistral-7B

MLflow
