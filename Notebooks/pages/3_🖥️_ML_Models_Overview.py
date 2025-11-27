import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# =========================================================
# ✅ BULLETPROOF BASE PATH (POINTS TO REPO ROOT)
# Final-project-DEPI/
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]

# =========================================================
# ✅ LOAD TEST DATA SAFELY (FROM REPO ROOT)
# =========================================================
X_test_path = BASE_DIR / "X_test.csv"
y_test_path = BASE_DIR / "y_test.csv"

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()


# =========================================================
# ✅ Helper: Display Confusion Matrix + Metrics
# =========================================================
def show_model_results(model_name, model_filename):
    st.header(f"📌 {model_name} Results")

    # ✅ Load model safely from repo root
    model_path = BASE_DIR / model_filename
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plotly Confusion Matrix
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"{model_name} – Confusion Matrix",
        labels={"x": "Predicted Label", "y": "True Label", "color": "Count"},
    )

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Not Churn", "Churn"]),
        yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Not Churn", "Churn"]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show matrix table
    st.subheader("Confusion Matrix Values")
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"],
    )
    st.table(cm_df)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


# =========================================================
# ✅ RUN ALL BASE MODELS
# =========================================================
show_model_results("Logistic Regression", "log_reg_model.joblib")
show_model_results("Random Forest Classifier (Base)", "rf_model.joblib")
show_model_results("XGBoost (Base)", "xgb_model.joblib")

st.markdown("---")

# =========================================================
# ✅ TUNED MODELS
# =========================================================
st.header("🏆 Tuned Models", divider=True)

show_model_results("Random Forest Classifier (Tuned)", "rf_best_model.joblib")
show_model_results("XGBoost (Tuned)", "xgb_best_model.joblib")

st.markdown("""
# **Customer Churn Model Tuning & Interpretation Report**

### The goal of this phase was to tune and interpret two machine learning models — **Random Forest** and **XGBoost** — to predict customer churn based on the available data.

## ✅ Random Forest (Tuned)
- Best CV Accuracy: **0.9309**
- Final Test Accuracy: **0.9321**
- Strong generalization with controlled overfitting

## ✅ XGBoost (Tuned)
- Best CV Accuracy: **0.9331**
- Final Test Accuracy: **0.9340**
- Slightly better than Random Forest due to boosting

### ✅ Final Choice: **XGBoost**
""")

st.markdown("""
## ✅ Business Impact
The model can reliably detect customers at risk of churn using:
- Tenure
- Monthly charges
- Contract type
- Customer service calls

This allows for **early intervention and targeted retention strategies**.
""")
