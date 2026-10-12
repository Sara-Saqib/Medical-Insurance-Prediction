"""
Medical Insurance Cost Prediction System – Streamlit web app
Course: Machine Learning Fundamentals (AIC354)

Loads 'medical_insurance_model.pkl' (trained in the Colab notebook) and predicts
the medical insurance charges (USD) of a person from 6 personal details.
"""

import os
import pickle

import pandas as pd
import streamlit as st

MODEL_FILE = "medical_insurance_model.pkl"
DATA_FILE = "insurance.csv"

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered")


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def retrain_model():
    """Fallback: rebuild the model from insurance.csv if the .pkl cannot be loaded
    (for example, because the scikit-learn version on the server is different)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    data = pd.read_csv(DATA_FILE).drop_duplicates()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    model = Pipeline([("preprocessor", preprocessor),
                      ("regressor", HistGradientBoostingRegressor(random_state=42))])
    model.fit(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES], data["charges"])
    return model


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    """Returns (model, note). 'note' is None when the saved model loaded normally."""
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f), None
    except Exception as error:  # missing file, version mismatch, corrupted pickle ...
        if os.path.exists(DATA_FILE):
            return retrain_model(), (
                f"The saved model could not be loaded ({type(error).__name__}), so the model "
                "was re-trained from insurance.csv (Histogram Gradient Boosting).")
        raise


model, load_note = load_model()

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    "Enter a person's details to estimate their **yearly medical insurance charges (USD)**. "
    "The model was selected using **10-fold cross validation** on the Medical Cost Personal Dataset."
)
if load_note:
    st.warning(load_note)

st.subheader("Personal details")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", min_value=18, max_value=64, value=35)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.5, step=0.1,
                          help="Body mass index. 18.5–24.9 is the 'normal' range.")
    children = st.slider("Number of children covered", min_value=0, max_value=5, value=0)
with col2:
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region (US)", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict charges", type="primary", use_container_width=True):
    person = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
    }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prediction = max(float(model.predict(person)[0]), 0.0)   # charges cannot be negative

    st.success("Prediction complete")
    st.metric("Estimated yearly medical charges", f"${prediction:,.2f}")

    with st.expander("Show the input sent to the model"):
        st.dataframe(person, hide_index=True)

with st.expander("About this app"):
    st.markdown(
        """
- **Task:** regression – predict `charges` from `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- **Algorithms compared:** ElasticNet, AdaBoost, Histogram Gradient Boosting, Bagging, MLP neural network.
- **Evaluation:** 10-fold cross validation on 100 % of the data, ranked by RMSE.
- **Pipeline:** StandardScaler (numeric) + OneHotEncoder (categorical) + best algorithm, all in one saved file.
        """
    )

st.info("⚠️ This is a student machine-learning project. The result is a statistical estimate, "
        "**not** an official insurance quotation or medical/financial advice.")
