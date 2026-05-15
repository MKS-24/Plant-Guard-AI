---
title: Plant Guard AI
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---


# PlantGuard AI - Plant Disease Classifier

An advanced AI-powered plant disease classification system using Deep Learning (CNN) and Computer Vision. Features a real-time dashboard with Sobel/Laplacian filter visualizations, high-fidelity accuracy charts, and multi-class disease detection. 

## Features
- **Real-time Inference:** Classify plant diseases from images with high confidence.
- **Computer Vision Filters:** Visualize Sobel and Laplacian edge detection filters.
- **Interactive Dashboard:** Beautifully designed responsive UI for all devices.
- **Model Performance:** Detailed training history and confusion matrix visualization.

## Tech Stack
- **Backend:** Flask, TensorFlow/Keras, OpenCV
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript (ES6+), Chart.js
- **Deployment:** Hugging Face Spaces (Docker)

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Khuzi13/plant-guard-ai.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:7860` in your browser.
