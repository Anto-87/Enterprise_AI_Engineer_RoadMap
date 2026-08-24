import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
import pickle 
from pathlib import Path

# Get the directory where app.py exists
BASE_DIR = Path(__file__).resolve().parent

# Model directory
MODEL_DIR = BASE_DIR / "models"


# load the trained model
model = tf.keras.models.load_model(MODEL_DIR / 'ann_model.h5')

with open(MODEL_DIR / "label_encoder_gender.pkl","rb") as f:
    label_encoder_gender = pickle.load(f)

with open(MODEL_DIR / "onehot_encoder_geo.pkl","rb") as f:
    label_encoder_geo = pickle.load(f)

with open(MODEL_DIR / "scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Input features
gender = st.selectbox("Gender", label_encoder_gender.classes_)
geo = st.selectbox("Geography", label_encoder_geo.categories_[0])
age = st.number_input("Age", min_value=0, max_value=100)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure=st.slider("Tenure", min_value=0, max_value=10)
num_of_products=st.slider("Number of Products", min_value=0, max_value=4)
has_cr_card=st.selectbox("Has Credit Card", [0,1])
is_active_member=st.selectbox("Is Active Member", [0,1])


# prepare thee input data 
input_data = pd.DataFrame({
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Geography": [geo],
    "Age": [age],
    "Balance": [balance],
    "CreditScore": [credit_score],
    "EstimatedSalary": [estimated_salary],
    "Tenure": [tenure],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member]
})

# one hot encoder for Geography

geo_encoded = label_encoder_geo.transform([[geo]]).toarray()
geo_encoded_df =pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(["Geography"]))


# combine one hot encoded with input data 
input_data = pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)

input_data = input_data[scaler.feature_names_in_]

# scale the input data
input_data_scaled = scaler.transform(input_data)

# make the prediction
prediction = model.predict(input_data_scaled)
predicition_proba = prediction[0][0]

if predicition_proba > 0.5:
    st.write(f"Customer is likely to churn with a probability of {predicition_proba:.2f}")
else:
    st.write(f"Customer is likely to remain with the bank with a probability of {1-predicition_proba:.2f}")