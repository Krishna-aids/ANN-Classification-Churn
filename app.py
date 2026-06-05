import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

# Define the Streamlit app
st.title("Customer Churn Prediction")

# Create input fields for user data
Geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
Gender = st.selectbox("Gender", label_encoder_gender.classes_)
Age = st.slider("Age", 18, 92)
Balance = st.number_input("Balance", min_value=0.0)
CreditScore = st.number_input("Credit Score", min_value=0.0)
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0)
NumOfProducts = st.slider("Number of Products", 1, 4)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict"):
    # One-hot encode Geography
    geography_encoded = one_hot_encoder.transform([[Geography]])
    geo_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=one_hot_encoder.get_feature_names_out(['Geography'])
    )

    # Build input DataFrame — column order must match scaler's feature_names_in_:
    # ['CreditScore', 'Gender', 'Age', 'Balance', 'NumOfProducts',
    #  'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    #  'Geography_France', 'Geography_Germany', 'Geography_Spain']
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Gender': [label_encoder_gender.transform([Gender])[0]],
        'Age': [Age],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary],
    })

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Reorder columns to exactly match scaler fit order
    input_data = input_data[scaler.feature_names_in_]

    # Scale and predict
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **unlikely to churn**.")