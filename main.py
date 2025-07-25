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

def load_data(data_dir, n_mels=128, sr=16000, target_shape=(128, 32)):
    X = []
    y = []
    labels = {'clap': 0, 'noise': 1}
    for label, idx in labels.items():
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            spectrogram = preprocess_audio(file_path, n_mels=n_mels, sr=sr)
            spectrogram = resize_spectrogram(spectrogram, target_shape)
            X.append(spectrogram)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

data_dir = r'I:\clapping\clapdetection'
X, y = load_data(data_dir)

# Normalize and reshape data
X = (X - np.min(X)) / (np.max(X) - np.min(X))
X = X[..., np.newaxis]  # Add channel dimension

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),  # This layer flattens the output of the convolutional layers
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: clap and noise
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Assuming the input shape of your spectrograms is now (128, 32, 1)
input_shape = (128, 32, 1)
model = create_model(input_shape)

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('clap_noise_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('clap_noise_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite format and saved as 'clap_noise_model.tflite'")