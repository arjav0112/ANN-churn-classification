import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pickle
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encode_geo.pkl','rb') as file:  
    onehot_encode_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Customer Churn Prediction", page_icon=":guardsman:", layout="wide")
st.title("Customer Churn Prediction")

st.subheader("Predicting whether a customer will churn or not")

geography = st.selectbox("Select Geography", onehot_encode_geo.categories_[0])
gender = st.selectbox("Select gender", label_encoder_gender.classes_)
age = st.slider("Enter Age", 18,100,30)
tenure = st.slider("Enter Tenure", 0,10,5)
balance = st.number_input("Enter Balance",0.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 150000.0, 50000.0)
credit_score = st.number_input("Credit Score", 0, 850, 700) 

input_data = {
    'CreditScore' : [credit_score],
    'Gender' : label_encoder_gender.fit_transform([gender])[0],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
}

input_df = pd.DataFrame(input_data)

geo_encode = onehot_encode_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encode_df = pd.DataFrame(geo_encode, columns=onehot_encode_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_encode_df], axis=1)

input_scaled = scaler.transform(input_df)





if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction_prob = prediction[0][0]
    print(prediction_prob)

    if prediction_prob > 0.5:
        st.error(f"The customer is likely to churn with a probability of {prediction_prob * 100}%")
    else:
        st.success(f"The customer is likely to stay with a probability of {(1 - prediction_prob)*100}%")
    st.write("### Prediction Probability")