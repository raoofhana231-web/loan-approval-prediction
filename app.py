import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4fd, #ffffff);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .result-approved {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 2rem;
        border-radius: 15px;
        border-left: 8px solid #28a745;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #155724;
    }
    .result-rejected {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 2rem;
        border-radius: 15px;
        border-left: 8px solid #dc3545;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


try:
    model, scaler, encoders = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")
    st.info("Please run the training notebook first to generate .pkl files.")


st.markdown('<div class="main-header">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("""
    - **Algorithm**: Decision Tree
    - **Accuracy**: ~97.8%
    - **F1-Score**: ~98.2%
    - **Imbalance**: SMOTE
    - **Top Feature**: CIBIL Score
    """)


if model_loaded:
    with st.form("loan_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", 0, 10, 2, 1)
        with col2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col3:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        st.markdown("### 💰 Financial Information")
        col4, col5, col6 = st.columns(3)
        with col4:
            income_annum = st.number_input("Annual Income (₹)", 0, value=5000000, step=100000)
        with col5:
            loan_amount = st.number_input("Loan Amount (₹)", 0, value=15000000, step=100000)
        with col6:
            loan_term = st.number_input("Loan Term (years)", 1, 30, 10, 1)

        cibil_score = st.slider("CIBIL Score", 300, 900, 700, 1)

        st.markdown("### 🏠 Asset Information")
        col9, col10 = st.columns(2)
        with col9:
            residential_assets_value = st.number_input("Residential Assets (₹)", 0, value=5000000, step=100000)
            luxury_assets_value = st.number_input("Luxury Assets (₹)", 0, value=15000000, step=100000)
        with col10:
            commercial_assets_value = st.number_input("Commercial Assets (₹)", 0, value=3000000, step=100000)
            bank_asset_value = st.number_input("Bank Asset (₹)", 0, value=5000000, step=100000)

        submitted = st.form_submit_button("🔍 Predict Loan Status")

    if submitted:
        education_enc = encoders['education'].transform([education])[0]
        self_employed_enc = encoders['self_employed'].transform([self_employed])[0]

        total_assets = (residential_assets_value + commercial_assets_value +
                        luxury_assets_value + bank_asset_value)
        loan_to_income_ratio = loan_amount / (income_annum + 1)
        asset_to_loan_ratio = total_assets / (loan_amount + 1)

        input_data = pd.DataFrame({
            'no_of_dependents': [no_of_dependents],
            'education': [education_enc],
            'self_employed': [self_employed_enc],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value],
            'total_assets': [total_assets],
            'loan_to_income_ratio': [loan_to_income_ratio],
            'asset_to_loan_ratio': [asset_to_loan_ratio]
        })

        input_data = input_data[encoders['feature_columns']]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        st.markdown("## 📋 Prediction Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-approved">✅ LOAN APPROVED<br>'
                f'<span style="font-size:1.2rem;">Confidence: {probability[1]*100:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-rejected">❌ LOAN REJECTED<br>'
                f'<span style="font-size:1.2rem;">Confidence: {probability[0]*100:.2f}%</span></div>',
                unsafe_allow_html=True
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Approval Probability", f"{probability[1]*100:.2f}%")
        with col_b:
            st.metric("Rejection Probability", f"{probability[0]*100:.2f}%")

        st.progress(float(probability[1]))

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888;'>Made with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)
