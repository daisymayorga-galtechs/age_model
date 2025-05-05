# app.py
import streamlit as st
import joblib

# Load model
model = joblib.load("age_model.pkl")

st.title("Child or Adult Predictor")

age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)

if st.button("Check"):
    prediction = model.predict([[age]])[0]
    label = "Adult" if prediction == 1 else "Child"
    st.success(f"You are predicted to be an **{label}**.")
