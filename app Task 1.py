import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

st.title("Credit Scoring ML Model")
st.write("This app allows you to train models on the credit dataset and evaluate prformance.")

df = pd.read_csv("UCI_Credit_Card.csv")

# Cleaning the Data
# ---- 1. EDUCATION ----
# Original values: 1=graduate school, 2=university, 3=high school, 4=others
# Extra values: 0, 5, 6 -> should all be grouped into "others" (4)
df['EDUCATION'] = df['EDUCATION'].replace({0:4, 5:4, 6:4})

# ---- 2. MARRIAGE ----
# Original values: 1=married, 2=single, 3=others
# Extra value: 0 -> recode into "others" (3)
df['MARRIAGE'] = df['MARRIAGE'].replace({0:3})

# ---- 3. Repayment Status (PAY_0 ... PAY_6) ----
# Values: -2, -1, 0, 1, ... , 9
# Merge -2 and -1 into 0 ("no delay")
pay_features = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
for col in pay_features:
    df[col] = df[col].replace([-2, -1], 0)

df["DEBT_TO_CREDIT_RAIO"] = (df["BILL_AMT1"] + df["BILL_AMT2"] + df["BILL_AMT3"] + df["BILL_AMT4"] + df["BILL_AMT5"] + df["BILL_AMT6"]) / df["LIMIT_BAL"]
df["PAYMENT_CONSITENCY_METRIC"] = (df["PAY_AMT1"] + df["PAY_AMT2"] + df["PAY_AMT3"] + df["PAY_AMT4"] + df["PAY_AMT5"] + df["PAY_AMT6"]) / (df["BILL_AMT1"] + df["BILL_AMT2"] + df["BILL_AMT3"] + df["BILL_AMT4"] + df["BILL_AMT5"] + df["BILL_AMT6"])


st.subheader("Dataset Preview:")
st.dataframe(df.head())

if st.checkbox("Show full dataset"):
    st.dataframe(df)

X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]

st.subheader("Train/Test Split")
test_size = st.slider("Test Set Size (30 Recommended)", 10, 20, 30) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Remove Inf and nans
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Scale to help the classification algorithms
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if st.checkbox("Show Training Data"):
    st.dataframe(pd.DataFrame(X_train, columns=X.columns))
if st.checkbox("Show Testing Data"):
    st.dataframe(pd.DataFrame(X_test, columns=X.columns))

# Create the models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

st.subheader("Select Model")
model_choice = st.selectbox("Choose a Model", list(models.keys()))
model = models[model_choice]

with st.expander("Train & Evaluate Model"):
    if st.button("Train & Evaluate"):
        with st.spinner("Training..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            try:
                roc = roc_auc_score(y_test, y_prob)
            except:
                roc = "N/A"
            
            st.subheader("Model Performance Metrics")
            st.write(f"**Accuracy:** {acc:.2f}")
            st.write(f"**F1 Class 1:** {f1:.2f}")
            st.write(f"**Recall:** {recall:.2f}")
            st.write(f"**ROC-AUC:** {roc}")

            report = classification_report(y_test, y_pred, output_dict=True)
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).T)

            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', linewidths=1, linecolor='black')
            st.pyplot(fig)

if st.checkbox("Test with your own data"):

    st.subheader("Predict on New Input")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.text_input(f"Input {col}", placeholder=f"Enter {col.lower()}", value=(str(df[col].mean()) if col != "PAYMENT_CONSITENCY_METRIC" else str(df[col][0])))
    if st.button("Predict"):
        if all(user_input.values()):
            input_df = pd.DataFrame([user_input])
            input_df = input_df.apply(pd.to_numeric, errors='ignore')

            input_df = input_df.replace([np.inf, -np.inf], np.nan)
            input_df = input_df.fillna(df.mean())

            input_df = scaler.transform(input_df)

            if 'trained_model' not in st.session_state:
                model.fit(X_train, y_train)
                st.session_state.trained_model = model
            
            prediction = st.session_state.trained_model.predict(input_df)[0]
            st.success(f"Predicted Target: {prediction}")
        else:
            st.warning("Please fill out all input fields")
