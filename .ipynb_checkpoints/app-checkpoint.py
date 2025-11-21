import streamlit as st
import pandas as pd
import pickle

# Load your trained model
@st.cache_resource
def load_model():
    with open("best_model-1.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="Health", layout="centered")

st.title("Medical Insurance Cost Predictor")
st.markdown("**Accurate predictions using  Random Forest model**")
st.markdown("Built by **Adembesa Godfrey** • R² ≈ 0.87 • Trained on real medical data")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 35, help="Age of the person")
    bmi = st.number_input("BMI", 15.0, 50.0, 27.5, 0.1, help="Body Mass Index")
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.radio("Sex", ["female", "male"])
    smoker = st.radio("Smoker?", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
    # Prepare input exactly like training
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
    input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
    input_data['age_smoker_interaction'] = input_data['age'] * input_data['smoker']

    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    st.success(f"### Predicted Annual Charges: **${prediction:,.2f}**")
    
    if smoker == "yes":
        st.warning("Smoking increases cost by ~$25,000–$35,000 per year!")
    else:
        st.info("Non-smokers save thousands every year")

    st.balloons()

st.markdown("---")
st.caption("Model: Random Forest (400 trees, max_depth=10) | Feature: age × smoker interaction |Best GridSearchCV model")
st.caption("contact:call +254707516308")
st.caption("contact:call +254788625382")
st.caption("Email. godfreyimbindi@gmail.com")
