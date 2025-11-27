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


# =========================================================
# ✅ RANDOM FOREST (STATIC SHOWCASE - REAL RESULTS)
# =========================================================
st.header("📌 Random Forest Classifier Results")

# ---- CONFUSION MATRIX IMAGE ----
st.subheader("Confusion Matrix")

try:
    st.image("rf_model.png", caption="Random Forest – Confusion Matrix", use_container_width=True)
except:
    st.info("Random Forest confusion matrix image not found.")

# ---- CLASSIFICATION METRICS (REAL VALUES) ----
st.subheader("Performance Metrics")

rf_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Value": [0.9352, 0.8978, 0.9972, 0.9449, 0.9533]
})

st.dataframe(rf_metrics, use_container_width=True)

# ---- CONFUSION MATRIX VALUES (REAL DATA) ----
st.subheader("Confusion Matrix Values")

rf_cm = pd.DataFrame(
    [[38372, 6387],
     [157, 56126]],
    columns=["Predicted Not Churn", "Predicted Churn"],
    index=["Actual Not Churn", "Actual Churn"]
)

st.table(rf_cm)

# ---- INTERPRETATION ----
st.markdown("""
✅ **Random Forest Model Interpretation:**

- The model achieved a **high accuracy of 93.52%**, confirming strong overall prediction reliability.
- The **very high recall of 99.72%** indicates that the model is extremely effective at detecting churned customers.
- The **ROC-AUC score of 95.33%** demonstrates excellent class separation capability.
- The low number of false negatives (**157 only**) is especially valuable for business churn prevention.

⚠️ The slightly lower **precision (89.78%)** indicates that a portion of non-churn customers are sometimes flagged as churn, which is acceptable in retention-focused strategies.
""")



show_model_results("XGBoost (Base)", "xgb_model.joblib")

st.markdown("---")

st.markdown("---")

st.header("🏆 Final Model", divider=True)
show_model_results("XGBoost (Tuned)", "xgb_best_model.joblib")


st.markdown("""
#  **Customer Churn Model Tuning & Interpretation Report**

### The goal of this phase was to tune and interpret two machine learning models — **Random Forest** and **XGBoost** — to predict customer churn based on the available data. Both models were trained, optimized, and evaluated using cross-validation and test accuracy.

## For the **Random Forest model**, the best parameters found were:
### `max_depth = 15`, `max_features = 'sqrt'`, `min_samples_leaf = 1`, `min_samples_split = 5`, and `n_estimators = 100`.
### The model achieved a **best cross-validation accuracy of 0.9309** and a **final test accuracy of 0.9321**.
### This means the model performs very well, correctly predicting around 93% of the cases.
### The deep trees allow the model to capture complex relationships in the data, while the chosen split and leaf parameters help maintain generalization and avoid overfitting.

## For the **XGBoost model**, the best parameters were:
### `colsample_bytree = 1.0`, `learning_rate = 0.1`, `max_depth = 7`, `n_estimators = 100`, and `subsample = 1.0`.
### The model achieved a **best cross-validation accuracy of 0.9331** and a **final test accuracy of 0.9340**.
### XGBoost slightly outperformed Random Forest. Its performance advantage comes from its ability to handle complex feature interactions and reduce errors through boosting. The moderate learning rate allows the model to learn gradually and avoid overfitting.
""")

st.markdown("---")

st.markdown("""
### In general, both models achieved very strong performance, with XGBoost showing a small but consistent improvement. Therefore, XGBoost can be considered the final chosen model for predicting customer churn.

## From a business perspective, these results indicate that the model can effectively identify customers who are likely to leave the service. Factors such as **tenure, contract type, number of customer service calls, and monthly charges** are likely to be among the most influential in predicting churn. Companies can use these insights to design targeted retention strategies, such as offering discounts or personalized support to at-risk customers.
""")

