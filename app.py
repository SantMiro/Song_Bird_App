#### Bird Song Classification App###
### Author: Santiago Miro ###

''' This Flask app predicts the type of bird, out of five birds by the sound they make. It yields the prediction and the confidence level.
    The app requests the user to upload an audio file (.wav) and the app will obtain the MFCC array of the file.
    Once the MFCC array is extracted, it is resized to be input to a pre-trained CNN model trained for American Robin, Bewick's Wren,
    Northern Cardinal, Northern Mockingbird and the Song Sparrow. '''

#Import libraries
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__)
#Load pre-trained model
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
    
    audio_data, sr = librosa.load(file, sr=None) # Load uploaded audiofile
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13) # Extract MFCC
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0) # Normalize
    mfccs_normalized = np.expand_dims(mfccs_normalized, axis=-1)  # Add channel dimension
    mfccs_normalized = np.expand_dims(mfccs_normalized, axis=0) # Add number of samples dimension
    mfccs_normalized = np.resize(mfccs_normalized, (1, 13, 130, 1)) # Resize
    prediction = model.predict(mfccs_normalized) # Make prediction
    predicted_index = np.argmax(prediction) # Obtain predicted index
    confidence = np.max(prediction) # Get Confidende Level
        
    # Check if confidence is below threshold or if the predicted index is out of range
    if confidence < 0.85 or predicted_index >= len(classes):
        return render_template('result.html', prediction="Not possible to determine.",
                               confidence="Under 0.85.")
        
    predicted_class = classes[predicted_index] # Obtain predicted class
    confidence_level = confidence * 100 
    confidence_level = str(round(confidence_level,4)) + '%' 
    return render_template('result.html', prediction=predicted_class, confidence = confidence_level)
if __name__ == '__main__':
    app.run(debug=True)
