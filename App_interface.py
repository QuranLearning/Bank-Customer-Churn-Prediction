import streamlit as st
import pickle
import pandas as pd

# Load full pipeline
model = pickle.load(open("full_pipeline.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.write("Fill in the customer details below:")

# -----------------------------
# User Inputs
# -----------------------------

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", value=50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
salary = st.number_input("Estimated Salary", value=50000.0)

# Dropdowns (Categorical)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Friendly Dropdowns → Convert to Numbers
has_card_text = st.selectbox("Has Credit Card?", ["No", "Yes"])
has_card = 1 if has_card_text == "Yes" else 0

is_active_text = st.selectbox("Is Active Member?", ["No", "Yes"])
is_active = 1 if is_active_text == "Yes" else 0

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "creditscore": credit_score,
    "geography": geography,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "numofproducts": num_products,
    "hascrcard": has_card,
    "isactivemember": is_active,
    "estimatedsalary": salary
}


input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Exit Status"):
    
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result:")

    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to EXIT\n\nProbability: {proba:.2%}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability of exit: {proba:.2%}")