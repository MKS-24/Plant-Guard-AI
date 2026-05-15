import cv2
import numpy as np
import tensorflow as tf

# Class labels
CLASS_LABELS = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
]

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Load image
img = cv2.imread("plant_light.jpg")

# Convert BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img = cv2.resize(img, (64, 64))

# Normalize
img = img.astype(np.float32) / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)

# Get highest probability class
index = np.argmax(predictions)

# Results
label = CLASS_LABELS[index]
confidence = predictions[0][index] * 100

print("Prediction:", label)
print(f"Confidence: {confidence:.2f}%")