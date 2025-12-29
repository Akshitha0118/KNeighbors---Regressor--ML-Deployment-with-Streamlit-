import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.markdown("<h1>💼 Employee Salary Predictor (KNN)</h1>", unsafe_allow_html=True)

# ✅ LOAD CSV FROM PROJECT DIRECTORY
data = pd.read_csv("emp_sal.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

model = KNeighborsRegressor(
    n_neighbors=2,
    algorithm='brute',
    leaf_size=100,
    p=1,
    weights='distance'
)
model.fit(X, y)

level = st.slider("Select Experience Level", 1.0, 10.0, 6.5, 0.1)

if st.button("Predict Salary"):
    salary = model.predict([[level]])[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")

