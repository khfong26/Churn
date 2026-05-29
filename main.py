import streamlit as st 
import pandas as pd 
import numpy as np
import pickle
import json

# title of the Web App
st.title("Customer Churn Risk Score Predictor")
st.header("This application predicts the risk score associated with a customer leaving (cancelling subscription, stop purchasing goods/services, etc.)")
st.write("Specify input conditions (parameters)")

# 1. READ LOCAL CSV (No more AWS connections)
df = pd.read_csv("cleaned_data.csv")
if "Unnamed: 0" in df.columns:
    del df["Unnamed: 0"]

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

def transform(df_input, freq_dict, cols):
    for c in cols:
        if c in freq_dict:
            subdict = freq_dict[c]
            df_input[f'per_{c}'] = df_input[c].map(subdict)
    return df_input

# 2. LOAD LOCAL JSON FREQUENCY DICTIONARY
with open("frequency_encoding.json", "r") as f:
    count_dict = json.load(f)

cols_to_transform = list(count_dict.keys())

# 3. LOAD LOCAL MODEL
with open("submit_model.pkl", "rb") as f:
    model = pickle.load(f)

x_input = user_inputs()
st.write('You selected:')
st.dataframe(x_input)

def predict(model, transformed):
    # XGBoost trained on y_train - 1, so we add 1 back to get the 1-5 score
    output = np.rint(model.predict(transformed)) + 1
    return output

if st.button("Predict"):
    transformed = transform(x_input, count_dict, cols_to_transform)
    
    try:
        # Reindexing ensures the dataframe columns perfectly match what XGBoost expects
        transformed = transformed.reindex(columns=model.feature_names_in_, fill_value=0)
        
        prediction = predict(model, transformed)
        final_score = int(prediction[0])
        
        st.subheader("Prediction based on your inputs:")
        st.write(f"The predicted Customer Churn Risk Score is: **{final_score}** (on a scale of 1-5)")
    except Exception as e:
        st.error(f"Error making prediction: {e}")