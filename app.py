from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__)
model = load_model('cnn_model.keras')

#Bird catalogue to classify from
classes = ['American Robin', "Bewick's Wren", 'Northern Cardinal', 
           'Northern Mockingbird', 'Song Sparrow']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    
    audio_data, sr = librosa.load(file, sr=None)
    #audio_data = audio_data[:22050]  # Assuming 1-second audio, adjust accordingly
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)
    mfccs_normalized = np.expand_dims(mfccs_normalized, axis=-1)  # Add channel dimension
    mfccs_normalized = np.expand_dims(mfccs_normalized, axis=0)
    mfccs_normalized = np.resize(mfccs_normalized, (1, 13, 130, 1)) 
    prediction = model.predict(mfccs_normalized)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)
        
    # Check if confidence is below threshold or if the predicted index is out of range
    if confidence < 0.85 or predicted_index >= len(classes):
        return render_template('result.html', prediction="Not possible to determine.",
                               confidence="Under 0.85.")
        
    predicted_class = classes[predicted_index]
    confidence_level = confidence * 100
    confidence_level = str(round(confidence_level,4)) + '%'
    return render_template('result.html', prediction=predicted_class, confidence = confidence_level)
if __name__ == '__main__':
    app.run(debug=True)
