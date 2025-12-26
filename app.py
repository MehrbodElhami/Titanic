import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Will you survive if you were among Titanic passengers? 🚢")

# ستون‌ها (ثابت تعریف می‌کنیم، وابسته به utils نیست)
columns = [
    'PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]

# لود امن مدل
@st.cache_resource
def load_model():
    return joblib.load("xgbpipe.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("❌ Model could not be loaded. Check versions or model file.")

# ورودی‌ها
passengerid = st.text_input("Passenger ID", "8585")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
name = st.text_input("Passenger Name", "Soheil Tehranipour")
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Number of siblings/spouses", 0, 10, 0)
parch = st.slider("Number of parents/children", 0, 10, 0)
ticket = st.text_input("Ticket Number", "8585")
fare = st.number_input("Fare", 0.0, 1000.0, 50.0)
cabin = st.text_input("Cabin", "C52")
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# پیش‌بینی
if st.button("Predict"):
    if not model_loaded:
        st.warning("Model is not available.")
    else:
        row = [
            passengerid,
            int(pclass),
            name,
            sex,
            float(age),
            int(sibsp),
            int(parch),
            ticket,
            float(fare),
            cabin,
            embarked
        ]

        X = pd.DataFrame([row], columns=columns)

        try:
            pred = model.predict(X)[0]
            if pred == 1:
                st.success("✅ Passenger Survived")
            else:
                st.error("❌ Passenger Did Not Survive")
        except Exception as e:
            st.error("Prediction failed. Model preprocessing mismatch.")
