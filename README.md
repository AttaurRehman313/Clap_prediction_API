# Clap vs Noise Audio Classifier API

This project provides a Flask-based API to classify audio files as either "clap" or "noise" using a pre-trained deep learning model.

## Project Structure

- `flaskApi.py` - Flask API for audio classification.
- `main.py` - (Not shown, but typically used for running or testing the model).
- `test.py` - (Not shown, but typically used for unit or integration tests).
- `clap_noise_model.h5` - Pre-trained Keras model for audio classification.
- `claping.mp3` - Example audio file.
- `requirements.txt` - Python dependencies.


## Requirements

Install dependencies using pip:

```sh
pip install -r requirements.txt
```

## Usage

1. **Start the API:**

   ```sh
   python flaskApi.py
   ```

2. **Send a POST request to `/predict` with an audio file:**

   Example using `curl`:
   ```sh
   curl -X POST -F "file=@claping.mp3" http://127.0.0.1:5000/predict
   ```

   The API will respond with a JSON indicating whether the audio is a clap or noise.

## API Endpoint

- `POST /predict`
  - **Form Data:** `file` (audio file, e.g., .mp3, .wav)
  - **Response:** JSON with prediction result.

## Model

The model (`clap_noise_model.h5`) is a Keras model trained to distinguish between clap sounds and noise using mel-spectrogram features.

## Notes

- The API saves uploaded files temporarily in an `uploads` directory (make sure this directory exists or is writable).
- The API deletes the uploaded file after prediction.
- Error handling is implemented for file upload and prediction steps.

## License

MIT License (add your own
