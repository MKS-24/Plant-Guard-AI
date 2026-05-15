import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN warnings
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys

# Screenshot mein yehi 5 classes nazar aa rahi hain.
# Agar model ki classes ka order kuch aur ho, toh is list ko update kar lijiye ga.
CLASS_NAMES = [
    "Pepper_bell__Bacterial_spot", "Pepper_bell__healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

def predict_image(model_path, img_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    target_size = (model.input_shape[1], model.input_shape[2])
    
    print(f"Loading image {img_path} with target size {target_size}...")
    try:
        img = image.load_img(img_path, target_size=target_size)
        original_img = image.load_img(img_path) # For display
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    img_array = image.img_to_array(img)
    
    # NORMALIZATION: Scale pixel values to [0, 1] as usually done during training
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    print("Running prediction...")
    predictions = model.predict(img_array)[0]
    
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    # Text Output
    print("\n" + "="*30)
    print("     PREDICTION RESULTS     ")
    print("="*30)
    print(f"Predicted Class Index: {predicted_class_idx}")
    if CLASS_NAMES and len(CLASS_NAMES) > predicted_class_idx:
        print(f"Disease Class: {CLASS_NAMES[predicted_class_idx]}")
    print(f"Confidence: {confidence:.2%}")
    print("="*30)
    
    # Graphical Output (Like the screenshot)
    show_visualization(original_img, predictions, predicted_class_idx)
    
    return predicted_class_idx, confidence

def show_visualization(img, predictions, predicted_class_idx):
    classes = CLASS_NAMES if CLASS_NAMES else [f"Class {i}" for i in range(len(predictions))]
    
    plt.figure(figsize=(9, 4))
    
    # 1. Show the Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    
    pred_label = classes[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    # Title Color (Green for Healthy, Red for Disease)
    color = 'green' if 'healthy' in pred_label.lower() else 'red'
    plt.title(f"{pred_label} ({confidence:.1f}%)", color=color, fontweight='bold', fontsize=14)
    
    # 2. Show the Bar Chart
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(classes))
    bars = plt.barh(y_pos, predictions, align='center', color='gray')
    
    plt.yticks(y_pos, classes)
    plt.gca().invert_yaxis()  # Read top-to-bottom
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    # Highlight the predicted bar
    bars[predicted_class_idx].set_color(color)
    
    # Add percentage text on bars
    for i, v in enumerate(predictions):
        plt.text(v + 0.02, i, f"{v*100:.1f}%", color='black', va='center', fontsize=10)
        
    plt.xlim(0, 1.15) # Max probability is 1.0, plus some space for text
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
    else:
        img_path = input("Please enter the path to the image: ").strip()
        img_path = img_path.strip('"').strip("'")
        
    if not img_path:
        print("No image path provided.")
        sys.exit(1)
        
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)
    predict_image("plant_disease_model_zipped.keras", img_path)
