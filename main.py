import streamlit as st 
from st_files_connection import FilesConnection
import pandas as pd 
import numpy as np
# import scipy as sp
import pickle
import datetime as dt
import json
from xgboost import plot_importance
from xgboost.sklearn import XGBRegressor as XGBR
import boto3
from io import BytesIO

# title of the Web App
st.title("Customer Churn Risk Score Predictor")
st.header("This application predicts the risk score associated with a customer leaving (cancelling subscription, stop purchasing goods/services, etc.)")
st.write("Specify input conditions (parameters)")

# define connection and df
conn = st.connection('s3', type=FilesConnection)
df = conn.read("airtraffic/before_encoding.csv", input_format="csv", ttl=600)

# transform the user_input as we have been transforming the data as before
def user_inputs():
    """
    define inputs should a user input into the app using streamlit's functions

    be sure to check all steps where we changed outliers, cleaned up odd strings, 
    cleaned data, and ignored some variables from EDA and feature_importance

    Args:
        None

    Returns:
        df: dataframe containing a single data point (1 row) with relevant columns.

    """
    data = "fill_in"
    x_input = pd.DataFrame(data, index=[0])
    return df

def transform(df):
    """
    define probability ratio encoding and/or other encodings that you have done.

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

# load saved model
# show team how to make a bucket with the secret key, etc.
s3 = boto3.resource('s3')
bucket_name = "type_your_bucket_name_here"

with BytesIO() as data:
    s3.Bucket(bucket_name).download_fileobj("final_model.pkl", data)
    data.seek(0)    # move back to the beginning after writing
    model = pickle.load(data)

# A confirmation so the user knows what the input row looks like
x_input = user_inputs()
st.write('You selected:')
st.dataframe(x_input)

# Predict with the model 
def predict(model, x_input):
    output = np.exp(model.predict(x_input))-1
    return output

# design user interface
if st.button("Predict"):
    x = transform(x_input)
    prediction = predict(model, x)
    st.subheader("Prediction based on your inputs:")

    # here, define more informative statements, such as recommended actions, cautions, statistics you want to include, etc...
    st.write(f"...\n {prediction}\n")