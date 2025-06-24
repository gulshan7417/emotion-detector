import os
import numpy as np
import librosa
from keras.models import load_model
import streamlit as st
import tempfile
import streamlit.components.v1 as components

# Load trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Emotion_Audio.h5')
emotion_classifier = load_model(model_path, compile=False)

# Emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Feature extraction function
def extract_features(data, sample_rate):
    result = np.array([])
    result = np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
    stft = np.abs(librosa.stft(data))
    result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)))
    return result

# Predict emotion from audio file path
def predict_emotion(file_path):
    data, sr = librosa.load(file_path, sr=44100)
    features = extract_features(data, sr)
    features_resized = np.resize(features, (162, 1))
    input_data = np.reshape(features_resized, (1, 162, 1))
    predictions = emotion_classifier(input_data)[0]
    dominant_index = np.argmax(predictions)
    dominant_emotion = emotion_labels[dominant_index]
    return dominant_emotion, predictions

# Streamlit Web App
st.set_page_config(page_title="🎙️ Audio Emotion Classifier", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e3f2fd, #bbdefb);
    font-family: 'Segoe UI', sans-serif;
    color: #000000;
}
h1, h3, h2 {
    color: #0d47a1;
}
.stButton button {
    background-color: #1976d2;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    transition: background-color 0.3s ease;
}
.stButton button:hover {
    background-color: #1565c0;
}
.block-container {
    padding: 2rem;
    border-radius: 12px;
    background-color: #e3f2fd;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    color: #000000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.uploadedFile {
    display: none;
}
footer {
    visibility: hidden;
}
.stFileUploader {
    border: 2px dashed #90caf9;
    padding: 1.5em;
    border-radius: 10px;
    background-color: #e3f2fd;
    transition: background-color 0.3s ease;
    color: #000000;
    text-align: center;
}
.stFileUploader:hover {
    background-color: #bbdefb;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Smart Audio Emotion Classifier")
st.markdown("#### Upload a `.wav` file to detect emotion from speech.")

audio_file = st.file_uploader("🔊 Upload WAV Audio", type=["wav"])
submit = st.button("🚀 Analyze Emotion")

st.markdown("<div id='prediction-result'></div>", unsafe_allow_html=True)

if audio_file is not None and submit:
    with st.spinner("🔍 Processing audio and detecting emotion..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getbuffer())
            temp_path = tmp.name

        emotion_label, confidence_array = predict_emotion(temp_path)

        st.markdown(f"""
        <div style='margin-top: 2em; background: #bbdefb; padding: 2em; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center;'>
            <h2 style='color: #0d47a1;'>🎯 Detected Emotion</h2>
            <h1 style='color: #1976d2; font-size: 3.5em; margin-top: 0.5em;'>{emotion_label.upper()}</h1>
        </div>
        """, unsafe_allow_html=True)

        components.html("""
        <script>
            const anchor = window.parent.document.getElementById("prediction-result");
            if (anchor) {
                anchor.scrollIntoView({behavior: 'smooth'});
            }
        </script>
        """, height=0)
