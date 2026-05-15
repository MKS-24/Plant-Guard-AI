from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import base64
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

import keras
from keras.layers import Dense

# --- PRO HACK: Patch Dense layer to ignore 'quantization_config' ---
# This fixes the "Unrecognized keyword arguments passed to Dense" error
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init
# ----------------------------------------------------------------

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_model_zipped.keras')
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    # Try loading without compile to bypass some version issues
    model = keras.models.load_model(model_path, compile=False)

CLASS_NAMES = ["Pepper_bell__Bacterial_spot",
    "Pepper_bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"]

def apply_filters(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    if np.max(sobel) > 0:
        sobel = np.uint8(sobel / np.max(sobel) * 255)
    else:
        sobel = np.uint8(sobel)
        
    # Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    if np.max(laplacian) > 0:
        laplacian = np.uint8(laplacian / np.max(laplacian) * 255)
    else:
        laplacian = np.uint8(laplacian)
    
    # Encode filters to base64 to send to frontend
    _, sobel_buffer = cv2.imencode('.jpg', sobel)
    sobel_b64 = "data:image/jpeg;base64," + base64.b64encode(sobel_buffer).decode('utf-8')
    
    _, lap_buffer = cv2.imencode('.jpg', laplacian)
    lap_b64 = "data:image/jpeg;base64," + base64.b64encode(lap_buffer).decode('utf-8')
    
    return sobel_b64, lap_b64

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Generate computer vision filters
        sobel_b64, lap_b64 = apply_filters(img_bgr)
        
        # Prepare image for model inference
        target_size = (model.input_shape[1], model.input_shape[2])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx]) * 100
        
        if predicted_class_idx < len(CLASS_NAMES):
            disease_name = CLASS_NAMES[predicted_class_idx]
        else:
            disease_name = f"Class {predicted_class_idx}"

        probabilities = {}
        for i, prob in enumerate(predictions):
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            probabilities[class_name] = round(float(prob) * 100, 2)

        return jsonify({
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'filters': {
                'sobel': sobel_b64,
                'laplacian': lap_b64,
                'featureMap': lap_b64 # Placeholder map
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'disease': f"Python Error: {str(e)}",
            'confidence': 0,
            'probabilities': {},
            'filters': {
                'sobel': '',
                'laplacian': '',
                'featureMap': ''
            }
        }), 200 # Returning 200 so the frontend fetch doesn't throw a network error and displays the message


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)