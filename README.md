# 🎧 Smart Audio Emotion Classifier

A sleek Streamlit app powered by a Keras model to classify emotions from speech in WAV audio files. Drop in your audio, hit analyze, and discover the emotion!

---

## 🚀 Features

- Intuitive, minimal UI with a modern bluish theme  
- Supports `.wav` file uploads  
- Real-time emotion detection using a trained Keras model  
- Display of predicted emotion (uppercase) in a stylish centered card  

---

## 🧠 Model & Audio Processing

- **Model**: `Emotion_Audio.h5` trained to recognize eight emotions: `angry, calm, disgust, fear, happy, neutral, sad, surprise`  
- **Feature Extraction**: Using `librosa` to extract:
  - Zero Crossing Rate
  - Chroma STFT
  - MFCCs
  - RMS Energy
  - Mel Spectrogram
- Model produces softmax probabilities; highest‑scoring emotion is shown

---

## 🛠️ Project Structure
<img width="1445" alt="Screenshot 2025-06-25 at 4 47 58 AM" src="https://github.com/user-attachments/assets/7007c284-85b4-4a61-8019-1b85d509da9f" />


- `app.py`:
  - Loads model
  - Defines `extract_features(...)` and `predict_emotion(...)`
  - Hosts interactive Streamlit front-end
  - Applies CSS for modern, centered UI

---

## 📥 Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/username/repo.git
cd repo/backend

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install librosa keras streamlit numpy

# 4. Ensure `Emotion_Audio.h5` is in the same folder as app.py

# ▶️ Run the App
streamlit run app.py


