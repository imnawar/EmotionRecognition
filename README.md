# Emotion Recognition Using YOLO

This repository contains the code and resources for our research on emotion recognition using the YOLO (You Only Look Once) object detection algorithm. The research aims to leverage YOLO's real-time object detection capabilities for accurate and swift facial emotion detection in images.

## Overview

This project uses YOLO (You Only Look Once), a fast and accurate object detection algorithm, to recognize emotions from facial expressions in images. YOLO processes the entire image in one go, predicting bounding boxes and class probabilities efficiently.

### Methodology

1. **Preprocessing**:
   - Resizing images to a 224×224 resolution.
   - Normalization techniques to enhance model generalization.
   - Labeling data as `c center_x center_y width height`, where:
     - `c` refers to the emotion class.
     - `center_x` and `center_y` represent the coordinates of the center of the box.
     - `width` and `height` are the width and height of the box, respectively.
   - Labels are saved in `txt` format with coordinates' values normalized to 0–1.

2. **Dataset**:
   - The MMAFEDB dataset is annotated for the classification task.
   - The WIDERFace dataset is used for training the YOLO model for face detection.

3. **YOLO Model Training**:
   - A trained YOLO model is used to detect faces in the dataset.
   - Self-supervised learning techniques are leveraged to label the emotion dataset, MMAFEDB.
   - A secondary YOLO model is trained for emotion detection using the labeled dataset.

### Results

The performance of the YOLO face emotion detection model was evaluated on a comprehensive test set. The key evaluation metrics were:
- **Precision (P)**: 0.438
- **Recall (R)**: 0.613
- **mAP50**: 0.551
- **mAP50-95**: 0.517

These results indicate a competitive ability to detect and classify emotions with room for improvement in reducing false positives.

## Files

- `Process_Data.ipynb`: This notebook is used to download the MMAFEDB dataset from [Kaggle](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression) and preprocess it as described in the methodology.
- `EmotionRecognition_YOLO.ipynb`: This notebook contains the code for training the YOLO model for emotion recognition.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- YOLOv5

### Installation

1. Preprocess the Data:
    - Open and run Process_Data.ipynb to download and preprocess the MMAFEDB dataset.

2. Train the YOLO Model:
    - Open and run EmotionRecognition_YOLO.ipynb to train the YOLO model for emotion detection.

  
## Acknowledgments

We would like to thank the developers of YOLO and the contributors of the MMAFEDB and WIDERFace datasets.
