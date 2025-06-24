import os
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
from flask import Flask, request, jsonify
from keras.models import load_model

# Load trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Emotion_Audio.h5')
emotion_classifier = load_model(model_path, compile=False)

# Emotion label map
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

# Predict emotion from audio array
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
