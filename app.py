import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Alzheimer's Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    with open("Alzheimers_Random_Forest_Model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🧠 Alzheimer's Disease Risk Prediction")

st.write(
"""
This AI tool predicts the **risk of Alzheimer's disease**
based on cognitive and behavioral assessments.

⚠️ This tool is for **educational purposes only** and does not replace medical diagnosis.
"""
)

st.divider()

# -------------------------------------------------
# Patient Assessment Section
# -------------------------------------------------

st.header("Patient Assessment")

col1, col2 = st.columns(2)

with col1:

    FunctionalAssessment = st.slider(
        "Functional Assessment Score",
        0.0, 10.0, 5.0
    )

    ADL = st.slider(
        "ADL Score (Activities of Daily Living)",
        0.0, 10.0, 5.0
    )

with col2:

    MMSE = st.slider(
        "MMSE Score",
        0, 30, 15
    )

    MemoryComplaints = st.selectbox(
        "Memory Complaints",
        ["No", "Yes"]
    )

    BehavioralProblems = st.selectbox(
        "Behavioral Problems",
        ["No", "Yes"]
    )

# Encode categorical values

MemoryComplaints = 1 if MemoryComplaints == "Yes" else 0
BehavioralProblems = 1 if BehavioralProblems == "Yes" else 0

st.divider()

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------

if st.button("🔎 Predict Risk Level"):

    # Feature order MUST match training data
    features = np.array([[
        MMSE,
        FunctionalAssessment,
        MemoryComplaints,
        BehavioralProblems,
        ADL
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    st.subheader("🧠 Prediction Result")

    # Risk Levels

    if probability < 0.30:
        st.success("✅ Low Risk of Alzheimer's Disease")

    elif probability < 0.60:
        st.warning("⚠️ Moderate Risk of Alzheimer's Disease")

    else:
        st.error("🚨 High Risk of Alzheimer's Disease")

    # -------------------------------------------------
    # Probability Display
    # -------------------------------------------------

    st.write("### Risk Probability")

    st.metric(
        label="Risk Score",
        value=f"{probability*100:.2f}%"
    )

    st.progress(int(probability * 100))

    st.divider()

    # -------------------------------------------------
    # Clinical Interpretation
    # -------------------------------------------------

    st.subheader("Clinical Interpretation")

    if probability < 0.30:

        st.info(
            "Patient shows strong cognitive health indicators."
        )

    elif probability < 0.60:

        st.info(
            "Some symptoms suggest possible cognitive decline. "
            "Further clinical evaluation is recommended."
        )

    else:

        st.info(
            "Patient shows strong indicators of Alzheimer's risk. "
            "Immediate medical consultation is advised."
        )

    st.divider()

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    st.subheader("Model Feature Importance")

    feature_names = [
        "MMSE",
        "FunctionalAssessment",
        "MemoryComplaints",
        "BehavioralProblems",
        "ADL"
    ]

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    
    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )
    
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance in Alzheimer's Prediction")
    
    plt.gca().invert_yaxis()
    
    st.pyplot(fig)

st.divider()

st.caption("Developed using Machine Learning | Random Forest | Streamlit")
