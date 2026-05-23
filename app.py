import streamlit as st
import pandas as pd
import joblib

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Churn Predictor", layout="wide")

# Load your model and the list of expected columns
model = joblib.load('customer_churn_model.pkl')
expected_features = joblib.load('feature_names.pkl')


st.title("Churn Predictor")

st.markdown("---")

# dictionary to store user inputs
user_inputs = {}
service_cols = [
    'Phone Service',
    'Multiple Lines',
    'Married',
    'Online Security',
    'Online Backup',
    'Device Protection Plan',
    'Premium Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Streaming Music',
    'Senior Citizen',
    'Unlimited Data',
    'Under 30',
    'Dependents',
    'Referred a Friend',
    'Internet Service',
    'Paperless Billing'
]
gender=['Male','Female']
payment_method=['Bank Withdrawal', 'Credit Card', 'Mailed Check']
# Dynamically create inputs based on the column type
city_list = joblib.load('city_list.pkl')
missing_fields = []
yesno=['Yes','No']
for col in expected_features:
    # Logic to decide what type of input to show
    if col in ['Age', 'Number of Dependents', 'Zip Code', 'Population', 'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue', 'CLTV']:
        user_inputs[col] = st.number_input(col, value=0.0)
    elif col == 'City':
        user_inputs[col] = st.selectbox("City", city_list)
    elif col == 'Payment Method':
        user_inputs[col] = st.selectbox("Payment Method", payment_method,index=0)
    elif col=='Gender':
        user_inputs[col]=st.selectbox("Gender",gender,index=0)
    elif col in service_cols:
        user_inputs[col]=st.selectbox(col,yesno,index=1)
    else:
        # Default to a text input or checkbox for other fields
        user_inputs[col] = st.text_input(col)
    if user_inputs[col]== "" or user_inputs[col] is None:
        missing_fields.append(col)


if missing_fields:
    st.error(f"Please fill in: {', '.join(missing_fields)}")
else:
    # Now build the DataFrame using the dictionary
    input_df = pd.DataFrame([user_inputs])

if st.button("Predict Churn"):
    with st.spinner('Analyzing patterns...'):


        # 1. Get the probability (returns an array like [[prob_no_churn, prob_churn]])
        probs = model.predict_proba(input_df)
        
        # 2. Extract the churn probability (index 1)
        churn_prob = probs[0][1]
        
        # 3. Get the hard prediction (0 or 1)
        prediction = model.predict(input_df)
        
        # 4. Display result
        if prediction[0] == 1:
            st.error(f"Prediction: **Churn**")
            st.write(f"The model is **{churn_prob*100:.1f}%** sure about this.")
        else:
            st.success(f"Prediction: **No Churn**")
            # For No Churn, we show the probability of NOT churning (prob 0)
            stay_prob = probs[0][0]
            st.write(f"The model is **{stay_prob*100:.1f}%** sure about this.")

        # Optional: Add a progress bar for visual flair
        st.progress(float(churn_prob))
        st.balloons()