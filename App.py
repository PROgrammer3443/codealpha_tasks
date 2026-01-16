# Task 1: Credit Scoring Model — CodeAlpha Internship
# Dataset: UCI_Credit_Card.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.title("Credit Scoring Model")
st.write("Train **Logistic Regression**, **Random Forest** or **Desicion Tree** Models on Credit Datasets")

# -------------------- Load Data --------------------
df = pd.read_csv("UCI_Credit_Card.csv")

# -------------------- Prepare Features --------------------
# Target variable
y = df["default.payment.next.month"]

# Features
X = df.drop(columns=["default.payment.next.month"])

# -------------------- Clean Data --------------------
# Replace inf and -inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values with column means
X = X.fillna(X.mean())

# -------------------- Train-Test Split --------------------
test_size = st.slider("Test Set Size (33% Recommended)", 10, 40, 33) / 100
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# -------------------- Scale Features --------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- Choose and Train Models --------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Random Forest
forest = RandomForestClassifier(random_state=42, n_estimators=100)
forest.fit(X_train, y_train)

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": tree,
    "Random Forest": forest
}
st.subheader("Select Model")
model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

if st.button("Train & Evaluate"):
    with st.spinner("Training in progress..."):
        # -------------------- Evaluate Models --------------------
        y_pred = models[model_choice].predict(X_test)
        st.subheader(f"**{model_choice}**")
        st.write("**Accuracy**:", accuracy_score(y_test, y_pred))
        st.write("**Precision**:", precision_score(y_test, y_pred))
        st.write("**Recall**:", recall_score(y_test, y_pred))
        st.write("**F1-Score**:", f1_score(y_test, y_pred))
        st.write("**ROC-AUC**:", roc_auc_score(y_test, y_pred))

        st.subheader("ROC-AUC Curve")
        plt.clf()
        RocCurveDisplay.from_estimator(models[model_choice], X_test, y_test)
        st.pyplot(plt)


        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
        st.pyplot(fig)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_pred),
        }
        st.subheader("Model Performance Summary")
        st.dataframe(pd.DataFrame(metrics, index=["Score"]).T)

st.subheader("Compare All Models (Auto-Evaluation)")
comparison = {}
for name, model in models.items():
    preds = model.predict(X_test)
    comparison[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, preds)
    }
st.dataframe(pd.DataFrame(comparison).T)
