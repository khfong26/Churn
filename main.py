import streamlit as st 
from st_files_connection import FilesConnection
import pandas as pd 
import numpy as np
import pickle
import datetime as dt
import json
from xgboost import plot_importance
from xgboost.sklearn import XGBRegressor as XGBR
import boto3
from io import BytesIO
from io import StringIO

# title of the Web App
st.title("Customer Churn Risk Score Predictor")
st.header("This application predicts the risk score associated with a customer leaving (cancelling subscription, stop purchasing goods/services, etc.)")
st.write("Specify input conditions (parameters)")

# FIX 1: Updated connection syntax
conn = st.connection('s3', type=FilesConnection)
df = conn.read("churn-challenge/cleaned_data.csv", input_format="csv", ttl=600)
del df["Unnamed: 0"]

# transform the user_input as we have been transforming the data as before
def user_inputs():
    # numerical 
    age = st.slider("How old is the customer", min_value=1, max_value=80, step=1)
    days_since_last_login = st.slider("Days since last login", min_value=1, max_value=80, step=1)
    points_in_wallet = st.number_input("Wallet Points", min_value=0, max_value=1000)
    joining_date = st.number_input("Date joined")
    avg_time_spent = st.number_input("Average time spent")
    avg_frequency_login_days = st.selectbox("Average login days", df["avg_frequency_login_days"].unique())
    # categorical 
    membership_category = st.selectbox("Select Membership Category", df["membership_category"].unique())
    feedback = st.selectbox("Select Feedback", df["feedback"].unique())
    complaint_status = st.selectbox("Select Complaint Status", df["complaint_status"].unique())
    region_category = st.selectbox("Select Region Category", df["region_category "].unique())
    medium_of_operation = st.selectbox("Select Medium of Operation", df["medium_of_operation"].unique())
    preferred_offer_types = st.selectbox("Preferred Offer Types", df["preferred_offer_types"].unique())
    internet_option = st.selectbox("Select internet_option", df["internet_option"].unique())
    gender = st.selectbox("Gender", df["gender"].unique())
    used_special_discount = st.selectbox("Used Special Discount", df["used_special_discount"].unique())
    
    data = {
        'age': age,
        'days_since_last_login': days_since_last_login,
        'points_in_wallet': points_in_wallet,
        'joining_date': joining_date,
        'avg_time_spent': avg_time_spent,
        'avg_frequency_login_days': avg_frequency_login_days,
        'membership_category': membership_category,
        'feedback': feedback,
        'complaint_status': complaint_status,
        'region_category': region_category,
        'medium_of_operation': medium_of_operation,
        'preferred_offer_types': preferred_offer_types,
        'internet_option': internet_option,
        'gender': gender,
        'used_special_discount': used_special_discount}
    
    x_input = pd.DataFrame(data, index=[0])
    return x_input

# Function to read a CSV file from S3
def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    return df

def transform(df, freq_dict, cols):
    for c in cols:
        subdict = freq_dict[c]
        df[f'per_{c}'] = df[c].map(subdict)

bucket_name = "churn-challenge"

# FIX 2: Added the specific file name string
df = read_csv_from_s3(bucket_name, "cleaned_data.csv")

s3 = boto3.resource('s3')

# FIX 3: Uncommented the model loading sequence
with BytesIO() as data:
    s3.Bucket("churn-challenge").download_fileobj("submit_model.pkl", data)
    data.seek(0)    
    model = pickle.load(data)

count_dict = conn.read("churn-challenge/freq_dict.json", input_format="json", ttl=600)

x_input = user_inputs()
st.write('You selected:')
st.dataframe(x_input)

def predict(model, transformed):
    # FIX 4: Removed the "np." typo
    output = np.rint(model.predict(transformed))
    return output

# design user interface
if st.button("Predict"):
    transformed = transform(x_input)
    prediction = predict(model, transformed)
    st.subheader("Prediction based on your inputs:")
    st.write(f"...\n {prediction}\n")