import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape, Permute, TimeDistributed

# -----------------------
# STREAMLIT APP HEADER
# -----------------------
st.title("🎵 Emotion Recognition from Speech")
st.write("Train CNN or CNN+LSTM models on the **RAVDESS speech dataset** to classify emotions.")

# -----------------------
# EMOTIONS
# -----------------------
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

dataset_path = r"D:\Coding\Internship 3-Months\Task 2\Audio_Speech_Actors_01-24"
n_mfcc, max_len = 40, 173

# -----------------------
# FEATURE EXTRACTION
# -----------------------
@st.cache_data
def load_data(path):
    X, y = [], []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]
                emotion = emotions.get(emotion_code)
                if emotion:
                    y_audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
                    if mfcc.shape[1] < max_len:
                        mfcc = np.pad(mfcc, pad_width=((0,0),(0, max_len - mfcc.shape[1])), mode="constant")
                    else:
                        mfcc = mfcc[:, :max_len]
                    X.append(mfcc)
                    y.append(emotion)
    return np.array(X), np.array(y)

# -----------------------
# LOAD DATASET ONCE
# -----------------------
if "data_loaded" not in st.session_state:
    with st.spinner("Extracting MFCC features..."):
        X, y = load_data(dataset_path)
        st.session_state.X, st.session_state.y = X, y
        st.session_state.data_loaded = True

X, y = st.session_state.X, st.session_state.y
st.success(f"✅ Dataset loaded with {len(X)} samples")

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train/test split
test_size = st.slider("Test Set Size (20 recommended)", 10, 40, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------
# MODEL SELECTION
# -----------------------
st.subheader("⚙️ Select Model")
model_choice = st.selectbox("Choose a Model", ["CNN", "CNN + LSTM"])

if model_choice == "CNN":
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, max_len, 1)),
        MaxPooling2D((2, 2)), Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(len(emotions), activation='softmax')
    ])
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, max_len, 1)),
        MaxPooling2D((2, 2)), Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),

        Permute((2, 1, 3)),
        TimeDistributed(Flatten()),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(len(emotions), activation='softmax')
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# with st.expander("📋 Model Summary"):
#     stringlist = []
#     model.summary(print_fn=lambda x: stringlist.append(x))
#     st.text("\n".join(stringlist))


# -----------------------
# TRAINING
# -----------------------
epochs = st.slider("Epochs", 10, 50, 20)
batch_size = st.slider("Batch Size", 16, 64, 32)

if st.button("🚀 Train & Evaluate"):
    with st.spinner("Training in progress..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    st.success("✅ Training Complete")

    # Training curves
    st.subheader("📈 Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Acc')
    ax.plot(history.history['val_accuracy'], label='Val Acc')
    ax.legend()
    st.pyplot(fig)

    # Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    
    try:
        roc = roc_auc_score(y_test, y_pred_probs, average="weighted", multi_class="ovr")
    except:
        roc = "N/A"

    st.subheader("📊 Model Performance Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**ROC-AUC:** {roc}")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).T)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=le.classes_, yticklabels=le.classes_)
    st.pyplot(fig)
