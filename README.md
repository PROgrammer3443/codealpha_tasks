# CodeAlpha Machine Learning Tasks

This repository contains three independent machine learning tasks, each implemented as a standalone **Streamlit** application. Each task can be run separately using its own `App.py` file.

---

## Task 1: Credit Scoring Model

**Directory:** `CodeAlpha_CreditScoringModel`

### Objective
Predict an individual’s creditworthiness using historical financial data.

### Description
This task applies classification techniques to assess credit risk based on features such as income, debt, and payment history. Model performance is evaluated using standard metrics including Precision, Recall, F1-Score, and ROC-AUC.

### Dataset
- `UCI_Credit_Card.csv` (included in the directory)

### Run Instructions
```bash
cd CodeAlpha_CreditScoringModel
streamlit run App.py
Task 2: Emotion Recognition from Speech
Directory: CodeAlpha_EmotionRecognitionFromSpeech

Objective
Recognize human emotions (e.g., happy, angry, sad) from speech audio.

Description
The application extracts audio features such as MFCCs and uses deep learning techniques to classify emotions from speech recordings.

Required Dataset
You must manually download the dataset before running the application.

Download Audio_Speech_Actors_01-24.zip from:
https://zenodo.org/records/1188976

After downloading, extract the contents and place them in the directory structure expected by the application.

Run Instructions
bash
Copy code
cd CodeAlpha_EmotionRecognitionFromSpeech
streamlit run App.py
Task 3: Handwritten Character Recognition
Directory: CodeAlpha_HandwrittenCharacterRecognition

Objective
Identify handwritten digits or characters using image processing and deep learning.

Description
This task uses a convolutional neural network trained on handwritten digit data (MNIST). For optimal accuracy, digits should be written clearly with good spacing, consistent stroke thickness, and proper alignment, as the system performs best with clean and well-structured handwriting.

Files
App.py – Streamlit application

mnist_model.h5 – Pre-trained model

train.py – Training script

Run Instructions
bash
Copy code
cd CodeAlpha_HandwrittenCharacterRecognition
streamlit run App.py
Requirements
Common dependencies include:

Python 3.x

streamlit

numpy

pandas

scikit-learn

tensorflow / keras

librosa (Task 2)

pillow

Install required packages using pip as needed.
