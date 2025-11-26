import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------
# Load test data
# ------------------------------
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()


# =========================================================
# Helper: Display Confusion Matrix + Metrics
# =========================================================
def show_model_results(model_name, model_path):
    st.header(f"📌 {model_name} Results")

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Compute Confusion Matrix
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

    # Show values in a table
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
# Run all models sequentially
# =========================================================
show_model_results("Logistic Regression", "log_reg_model.joblib")
show_model_results("Random Forest Classifier (Base)", "rf_model.joblib")
show_model_results("XGBoost (Base)", "xgb_model.joblib")


st.markdown("---")

st.header("🏆 Tuned Models", divider=True)
show_model_results("Random Forest Classifier (tuned)", "rf_best_model.joblib")
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
