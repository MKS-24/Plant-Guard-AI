# Automated Plant Disease Classifier with Filter Visualization

## Project Structure
```
plant_disease/
├── 1_data_pipeline.py       # Data loading, augmentation, preprocessing
├── 2_filter_visualization.py # Sobel, Laplacian, Gaussian, manual kernels
├── 3_model.py               # CNN architecture from scratch
├── 4_train.py               # Training loop with callbacks
├── 5_evaluate.py            # Confusion matrix, metrics, annotation
├── 6_predict.py             # Single image prediction + filter grid
├── utils/helpers.py         # Shared utility functions
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download PlantVillage from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease

Use these 5 classes (folders):
- Tomato_healthy
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot

Place them in: data/PlantVillage/

## Run Order
```bash
python 1_data_pipeline.py      # Verify dataset + show samples
python 2_filter_visualization.py  # See filter outputs
python 4_train.py              # Train the model (~20 min on CPU)
python 5_evaluate.py           # Confusion matrix + metrics
python 6_predict.py --image path/to/leaf.jpg  # Predict any image
```
