import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# =========================================================
# ✅ AUTO-DETECT PROJECT ROOT
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # Final-project-DEPI

# =========================================================
# ✅ AUTO-FIND FILES ANYWHERE IN PROJECT
# =========================================================
def find_file(filename):
    matches = list(BASE_DIR.rglob(filename))
    if not matches:
        st.error(f"❌ File not found anywhere in the project: {filename}")
        st.stop()
    return matches[0]

# =========================================================
# ✅ LOAD TEST DATA SAFELY
# =========================================================
X_test_path = find_file("X_test.csv")
y_test_path = find_file("y_test.csv")

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

# =========================================================
# ✅ Helper: Display Confusion Matrix + Metrics
# =========================================================
def show_model_results(model_name, model_filename):
    st.header(f"📌 {model_name} Results")

    # ✅ Load model safely from anywhere
    model_path = find_file(model_filename)
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
# ==========================
