import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_audio(file_path, n_mels=128, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    log_spectrogram = np.log(spectrogram + 1e-9)  # add small number to avoid log(0)
    return log_spectrogram

def resize_spectrogram(spectrogram, target_shape):
    from skimage.transform import resize
    return resize(spectrogram, target_shape, mode='constant')

# Load the trained model
model = tf.keras.models.load_model('clap_noise_model.h5')

def predict_audio(file_path, model, n_mels=128, sr=16000, target_shape=(128, 32)):
    spectrogram = preprocess_audio(file_path, n_mels=n_mels, sr=sr)
    
    # Resize, normalize and reshape
    spectrogram = resize_spectrogram(spectrogram, target_shape)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = spectrogram[..., np.newaxis]  # Add channel dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(spectrogram)
    return np.argmax(prediction)

# Path to the mp3 file you want to predict
mp3_file_path = 'claping.mp3'  # Change this to the path of your MP3 file
prediction = predict_audio(mp3_file_path, model)

if prediction == 0:
    print("The audio is a clap sound.")
else:
    print("The audio is noise.")
