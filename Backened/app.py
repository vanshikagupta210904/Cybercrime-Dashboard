from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
from flask_cors import CORS

import json  # Import the json module

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3001"}})

# Load the model architecture from JSON
with open('Backened/model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("Backened/model_weights.h5")

# Define a route for making predictions
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
   
    file = request.files['file']
    
    # file = 'backened/New-recording-23.wav'
    if file:
        sound_signal, sample_rate = librosa.load(file, sr=None, res_type="kaiser_fast")  # Add sr=None
        mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
        mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
        mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
        result_array = loaded_model.predict(mfccs_features_scaled)
        print(result_array)
        result_classes = ["FAKE", "REAL"]
        result = np.argmax(result_array[0])
        print("Result:", result_classes[result])
        return jsonify({'predictions':  result_classes[result]})
    else:
        return "Invalid file"

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development purposes


