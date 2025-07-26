import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data(file_path):
    print("🔄 Loading data...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    print("\n🔄 Encoding features...")
    processed_df = df.copy()
    label_encoders = {}

    # ✅ Clean 'age' column: remove "years", then normalize
    # processed_df['age'] = (
    #     processed_df['age']
    #     .astype(str)
    #     .str.replace("years", "", case=False)
    #     .str.replace("yea", "", case=False)
    #     .str.replace("ye", "", case=False)
    #     .str.replace("y", "", case=False)
    #     .str.strip()
    # )

    # ✅ Replace invalid ages (numeric, empty, or unexpected) with UNKNOWN
    # valid_ages = [
    #     "18-24", "25-34", "35-44", "45-54",
    #     "55-64", "65-74", "75-84", "85-89", "ABOVE 90"
    # ]
    # processed_df['age'] = processed_df['age'].apply(
    #     lambda x: x if x in valid_ages else "UNKNOWN"
    # )

    #print("DEBUG - Unique ages after cleaning:", processed_df['age'].unique())

    
    
    # ✅ Gender: convert numeric to M/F
    # processed_df['gender'] = processed_df['gender'].apply(
    # lambda x: 'M' if pd.notna(x) and str(int(float(x))) == '1' else 'F'
    # )
    # # ✅ Predefined categories
    # predefined_categories = {
        
    #     "gender": ["M", "F"],
    #     "age": [
    #         "18-24", "25-34", "35-44", "45-54",
    #         "55-64", "65-74", "75-84", "85-89", "ABOVE 90", "UNKNOWN"
    #     ]
    # }

    categorical_columns = ['antibiotic']

    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(processed_df[col].unique())  # Only for antibiotic
        processed_df[f'{col}_encoded'] = le.transform(processed_df[col])
        label_encoders[col] = le
        print(f"✓ Encoded {col} ({len(le.classes_)} classes)")
            
    feature_columns = [
        'median_heartrate', 'median_resprate', 'median_temp',
        'median_sysbp', 'median_diasbp',
        'median_wbc', 'median_hgb', 'median_plt',
        'median_na', 'median_hco3', 'median_bun', 'median_cr'
        #'age_encoded', 'gender_encoded'
    ]

    return processed_df, feature_columns, label_encoders
