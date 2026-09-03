
import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("rental_scam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

features = [
    "LISTING_KIND",
    "LISTING_CITY",
    "LISTING_PRICE",
    "LOGIN_COUNTRY_CODE",
    "LISTING_COUNTRY_CODE",
    "LISTING_REGISTRATION_POSSIBLE",
    "ADVERTISER_COMPLETENESS_SCORE",
    "MANAGED_ACCOUNT",
    "HAS_PROFILE_PIC",
    "BROWSER",
    "OS",
    "IS_ARCHIVED",
    "PRICE_LOG",
    "LOGIN_COUNTRY_MISSING"
]

categorical_columns = [
    "LISTING_CITY",
    "BROWSER",
    "OS",
    "LOGIN_COUNTRY_CODE",
    "LISTING_COUNTRY_CODE"
]

st.set_page_config(
    page_title="Rental Scam Detector",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Rental Scam Detection System")
st.write("Enter rental listing details to check whether the listing may be a scam.")

st.divider()

listing_kind = st.number_input(
    "Listing Kind",
    min_value=0,
    value=1
)

listing_city = st.text_input(
    "City",
    "London"
)

listing_price = st.number_input(
    "Monthly Rent",
    min_value=0.0,
    value=800.0
)

login_country_code = st.text_input(
    "Login Country Code",
    "GB"
)

listing_country_code = st.text_input(
    "Listing Country Code",
    "GB"
)

registration_possible = st.selectbox(
    "Registration Possible",
    [0, 1]
)

advertiser_score = st.number_input(
    "Advertiser Completeness Score",
    min_value=0.0,
    value=50.0
)

managed_account = st.selectbox(
    "Managed Account",
    [0, 1]
)

has_profile_pic = st.selectbox(
    "Profile Picture Available",
    [0, 1]
)

browser = st.text_input(
    "Browser",
    "Chrome"
)

os = st.text_input(
    "Operating System",
    "Windows"
)

is_archived = st.selectbox(
    "Is Archived",
    [0, 1]
)

st.divider()

if st.button("🔍 Check Rental Listing", use_container_width=True):

    input_data = pd.DataFrame([{
        "LISTING_KIND": listing_kind,
        "LISTING_CITY": listing_city,
        "LISTING_PRICE": listing_price,
        "LOGIN_COUNTRY_CODE": login_country_code,
        "LISTING_COUNTRY_CODE": listing_country_code,
        "LISTING_REGISTRATION_POSSIBLE": registration_possible,
        "ADVERTISER_COMPLETENESS_SCORE": advertiser_score,
        "MANAGED_ACCOUNT": managed_account,
        "HAS_PROFILE_PIC": has_profile_pic,
        "BROWSER": browser,
        "OS": os,
        "IS_ARCHIVED": is_archived
    }])

    input_data["PRICE_LOG"] = np.log1p(
        input_data["LISTING_PRICE"]
    )

    input_data["LOGIN_COUNTRY_MISSING"] = (
        input_data["LOGIN_COUNTRY_CODE"].isnull().astype(int)
    )

    for col in categorical_columns:
        input_data[col] = (
            input_data[col]
            .fillna("Unknown")
            .astype(str)
        )

        le = label_encoders[col]
        value = input_data[col].iloc[0]

        if value in le.classes_:
            input_data[col] = le.transform([value])
        else:
            input_data[col] = 0

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 POTENTIAL SCAM")
    else:
        st.success("✅ LIKELY LEGITIMATE")

    st.metric(
        "Scam Probability",
        f"{probability * 100:.2f}%"
    )
