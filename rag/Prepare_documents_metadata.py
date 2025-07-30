import pandas as pd

# Load your filtered data
df = pd.read_csv("cleaned_antibiotic_data.csv")

# Ensure required columns exist
required_cols = ['order_proc_id_coded', 'organism', 'antibiotic', 'antibiotic_class', 'culture_description', 'hosp_ward_ICU', 'age']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Prepare docs and metadata
documents, metadatas, ids = [], [], []

for i, row in df.iterrows():
    text = (
        f"Patient case: {row['culture_description']} culture, organism {row['organism']}. "
        f"Antibiotic tested: {row['antibiotic']}, class: {row['antibiotic_class']}. "
        f"ICU: {'Yes' if row['hosp_ward_ICU'] == 1 else 'No'}, age: {row['age']}."
    )
    documents.append(text)
    metadatas.append({
        "order_id": str(row['order_proc_id_coded']),
        "organism": row['organism'],
        "antibiotic_class": row['antibiotic_class'],
        "icu": row['hosp_ward_ICU'],
        "age": row['age']
    })
    ids.append(f"{row['order_proc_id_coded']}_{i}")  # unique ID

print(f"✅ Prepared {len(documents)} documents for indexing")
