import streamlit as st
import pandas as pd
import joblib


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# 2. LOAD SAVED MODEL
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")


model_bundle = load_model()

model = model_bundle["model"]
threshold = model_bundle["threshold"]


# 3. APPLICATION TITLE
st.title("📊 Customer Churn Prediction System")

st.write(
    "An AI-powered application that predicts "
    "whether a telecom customer is likely to leave."
)

st.divider()


# 4. CUSTOMER INFORMATION FORM
st.header("Enter Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submit = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


# 5. MAKE PREDICTION
if submit:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_data = pd.DataFrame([customer_data])

    input_data = input_data[model_bundle["features"]]

    # Get churn probability
    probability = model.predict_proba(input_data)[0][1]

    # Apply selected threshold
    prediction = int(probability >= threshold)

    st.divider()
    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.error("⚠️ This customer is likely to leave!")

        st.subheader("Suggested Retention Actions")

        st.write(
            "- Offer a suitable loyalty discount."
        )

        st.write(
            "- Contact the customer to understand their concerns."
        )

        st.write(
            "- Consider offering a long-term contract."
        )

    else:

        st.success("✅ This customer is likely to stay!")

        st.write(
            "Continue providing good customer service "
            "and monitor customer satisfaction."
        )


# 6. FOOTER
st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Machine Learning Project using Python and Scikit-learn"
)
