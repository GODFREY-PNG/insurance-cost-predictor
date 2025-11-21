# insurance_predictor.py
#  Realistic & Accurate Predictions
#  Built by Adembessa Godfrey Imbindi

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder

# ==============================
# 1. DATA PREPARATION 
# ==============================
def prepare_data(csv_path="insurance.csv"):
    df = pd.read_csv(csv_path)
    
    # preprocessing
    df["smoker"] = df["smoker"].map({'yes': 1, 'no': 0})
    df["sex"] = df["sex"].map({'male': 0, 'female': 1})
    
    # feature enginearing
    df['age_smoker_interaction'] = df['age'] * df['smoker']
    #spliting
    features = ["age", "sex", "bmi", "children", "smoker", "region", "age_smoker_interaction"]
    X = df[features]
    y = df["charges"]
    
    print(f"Data loaded: {df.shape[0]} rows")
    print("Features used:", features)
    return X, y

# ==============================
# 2. TRAIN AND SAVE MODEL
# ==============================
def train_and_save_model(csv_path="insurance.csv", model_path="best_model-1.pkl"):
    X, y = prepare_data(csv_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    )
    
    print("Training the final model...")
    model.fit(X_train, y_train.values.ravel())
    
    print(f"Train R²: {model.score(X_train, y_train):.4f}")
    print(f"Test  Test R²: {model.score(X_test, y_test):.4f}")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved as '{model_path}' → Ready for deployment!")

# ==============================
# 3. PREDICTION FUNCTION 
# ==============================
def predict_insurance_charges(age, sex, bmi, children, smoker, region,
                              model_path="best_model-1.pkl"):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Create input DataFrame
    new_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Apply SAME preprocessing as training
    new_data['smoker'] = new_data['smoker'].map({'yes': 1, 'no': 0})
    new_data['sex'] = new_data['sex'].map({'male': 0, 'female': 1})
    
    # new data feature engineering
    new_data['age_smoker_interaction'] = new_data['age'] * new_data['smoker']
    
    prediction = model.predict(new_data)[0]
    return round(float(prediction), 2)

# ==============================
# 4. RUN ONCE TO TRAIN
# ==============================
if __name__ == "__main__":
    if not os.path.exists("insurance.csv"):
        print("insurance.csv not found! Place it in this folder.")
    else:
        train_and_save_model()
        
        print("\nRealistic Example Predictions:")
        print("→ 30yo non-smoker female, BMI 25.0 → $",
              predict_insurance_charges(30, "female", 25.0, 0, "no", "southeast"))
        print("→ 45yo smoker male, BMI 30.0 → $",
              predict_insurance_charges(45, "male", 30.0, 2, "yes", "northwest"))
        print("→ 55yo smoker, BMI 33 → $",
              predict_insurance_charges(55, "male", 33.0, 0, "yes", "southeast"))