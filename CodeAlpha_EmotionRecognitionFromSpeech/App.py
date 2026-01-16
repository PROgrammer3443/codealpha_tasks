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

from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Permute, TimeDistributed
)

# -----------------------
# STREAMLIT APP HEADER
# -----------------------
st.title("Emotion Recognition from Speech")
st.write("Train CNN or CNN+LSTM models on the **RAVDESS speech dataset** to classify emotions.")

# -----------------------
# EMOTIONS
# -----------------------
emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

dataset_path = r"C:\Users\shara\Documents\Coding\CodeAlpha Internships\ML Internship\Task 2\Audio_Speech_Actors_01-24"
n_mfcc, max_len = 40, 173

# -----------------------
# FEATURE EXTRACTION
# -----------------------
@st.cache_data
def load_data(path):
    """
    Extract MFCC features and emotion labels from the RAVDESS dataset.
    Includes progress bar and error handling.
    """
    X, y = [], []
    audio_files = []

    # Collect all .wav file paths first
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    total_files = len(audio_files)
    progress = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(audio_files):
        try:
            file_name = os.path.basename(file_path)
            emotion_code = file_name.split("-")[2]
            emotion = emotions.get(emotion_code)
            if emotion:
                # Load and process audio
                y_audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)

                # Pad or truncate to fixed length
                if mfcc.shape[1] < max_len:
                    mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
                else:
                    mfcc = mfcc[:, :max_len]

                X.append(mfcc)
                y.append(emotion)

            # Update progress bar and text
            progress.progress((i + 1) / total_files)
            status_text.text(f"Processed {i + 1} / {total_files} files")

        except Exception as e:
            st.warning(f"⚠️ Error processing {file_path}: {e}")

    status_text.text("✅ Feature extraction complete!")
    return np.array(X), np.array(y)


# -----------------------
# LOAD DATASET
# -----------------------
if "data_loaded" not in st.session_state:
    with st.spinner("Extracting MFCC features..."):
        X, y = load_data(dataset_path)
        st.session_state.X, st.session_state.y = X, y
        st.session_state.data_loaded = True

X, y = st.session_state.X, st.session_state.y
st.success(f"✅ Dataset loaded with {len(X)} samples")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split train/test
test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=42)

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------
# MODEL SELECTION
# -----------------------
st.subheader("Select Model")
model_choice = st.selectbox("Choose a Model", ["CNN", "CNN + LSTM"])

if model_choice == "CNN":
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(n_mfcc, max_len, 1)),
        MaxPooling2D((2,2)), Dropout(0.3),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)), Dropout(0.3),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)), Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(len(emotions), activation='softmax')
    ])
else:
    # CNN + LSTM fix: Permute + TimeDistributed Flatten
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(n_mfcc, max_len, 1)),
        MaxPooling2D((2,2)), Dropout(0.3),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)), Dropout(0.3),
        Permute((2,1,3)),           # swap axes to (time_steps, features)
        TimeDistributed(Flatten()), # flatten features per time step
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(len(emotions), activation='softmax')
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# -----------------------
# TRAINING
# -----------------------
epochs = st.slider("Epochs", 10, 50, 20)
batch_size = st.slider("Batch Size", 16, 64, 32)

if st.button("Train & Evaluate"):
    with st.spinner("Training in progress..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    st.success("Training Complete")

    # Training curves
    st.subheader("Training History")
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
    except Exception as e:
        roc = f"NA ({e})"

    st.subheader("Model Performance Metrics")
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=le.classes_, yticklabels=le.classes_)
    st.pyplot(fig)

