#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:09:08 2024

@author: deepak
"""
from io import BytesIO
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load and compile the model
model = load_model('vegetable_freshness_model1.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Read the uploaded image into memory
        img_bytes = file.read()
        img = BytesIO(img_bytes)
        
        # Load and preprocess the image
        test_image = image.load_img(img, target_size=(128, 128))
        test_image = image.img_to_array(test_image) / 255.0  # Normalize pixel values
        test_image = np.expand_dims(test_image, axis=0)
        
        # Predict the result
        result = model.predict(test_image)
        print("Raw model prediction:", result)  # Debugging output
        
        # Adjust threshold if needed
        if(result[0][0]>=0.5):
            prediction="Not Fresh"
        else:
            prediction="Fresh"
        print("Predicted Class:", prediction)  # Debugging output
        
        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    