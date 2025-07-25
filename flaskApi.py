import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
from skimage.transform import resize
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('clap_noise_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_audio(file_path, n_mels=128, sr=16000):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        log_spectrogram = np.log(spectrogram + 1e-9)  # add small number to avoid log(0)
        return log_spectrogram
    except Exception as e:
        raise RuntimeError(f"Error processing audio file: {e}")

def resize_spectrogram(spectrogram, target_shape):
    try:
        return resize(spectrogram, target_shape, mode='constant')
    except Exception as e:
        raise RuntimeError(f"Error resizing spectrogram: {e}")

def predict_audio(file_path, model, n_mels=128, sr=16000, target_shape=(128, 32)):
    try:
        spectrogram = preprocess_audio(file_path, n_mels=n_mels, sr=sr)
        
        # Resize, normalize and reshape
        spectrogram = resize_spectrogram(spectrogram, target_shape)
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        spectrogram = spectrogram[..., np.newaxis]  # Add channel dimension
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
        
        # Predict
        prediction = model.predict(spectrogram)
        return np.argmax(prediction)
    except Exception as e:
        raise RuntimeError(f"Error predicting audio file: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Make prediction
        prediction = predict_audio(file_path, model)
        
        # Clean up the saved file
        os.remove(file_path)
        
        if prediction == 0:
            result = "The audio is a clap sound."
        else:
            result = "The audio is noise."
        
        return jsonify({"prediction": result})
    
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred. {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
