# Counterfeit IC Detection

## Overview

Counterfeit Integrated Circuits (ICs) pose a significant threat to industries such as electronics manufacturing, telecommunications, and aerospace, potentially resulting in unreliable products, increased costs, and even safety hazards. This project presents a comprehensive deep learning-based approach for detecting counterfeit ICs using Convolutional Neural Networks (CNNs). The solution leverages image-based analysis, data augmentation, and custom model design to achieve high detection performance even on relatively small datasets.

---

## Table of Contents

- [Overview](#overview)
- [Background & Motivation](#background--motivation)
- [Methodology](#methodology)
  - [Dataset & Data Preparation](#dataset--data-preparation)
  - [Data Augmentation](#data-augmentation)
  - [Model Architectures](#model-architectures)
    - [AlexNet](#alexnet)
    - [VGG16](#vgg16)
    - [Custom CNN Model](#custom-cnn-model)
  - [Training Procedure](#training-procedure)
- [Results & Evaluation](#results--evaluation)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

---

## Background & Motivation

Counterfeit ICs undermine the integrity of global supply chains. Traditional detection methods can be expensive, time-consuming, and require expert knowledge or destructive testing. Leveraging deep learning and computer vision offers an automated, scalable, and non-destructive solution for counterfeit detection by analyzing subtle visual features in IC images.

---

## Methodology

### Dataset & Data Preparation

- **Source**: The dataset consists of 85 high-resolution images of integrated circuits, classified as 50 "approved" and 35 "counterfeit".
- **Image Preprocessing**:
  - All images are resized to 960x720 (HxW, 3 channels).
  - Conversion to RGB and tensor representations is handled via OpenCV and torchvision.
  - Binary labels are assigned based on the filename prefix (e.g., 'A' for approved).
- **Splitting**:
  - Stratified splitting ensures class balance in train/test sets (80/20 split).

### Data Augmentation

Given the limited dataset size, extensive augmentation is performed to enhance generalizability:
- **Techniques**: Random rotations (up to 30 degrees), horizontal/vertical flips, and transformation to tensors.
- **Resulting Size**: Each training image is augmented 5 times, expanding the training set from 68 to 408 images, for a total of 430.

### Model Architectures

#### AlexNet

- **Structure**: Classic CNN architecture with stacked convolutional, pooling, and fully connected layers.
- **Training Performance**: 
  - Training/Validation Accuracy: ~58.8%
  - AlexNet proved too simple to capture the complex visual features needed for reliable detection.

#### VGG16

- **Approach**: Transfer learning using torchvision’s pre-trained VGG16.
- **Modifications**:
  - Classifier layer adapted for binary classification.
  - All feature extraction layers are frozen.
- **Performance**:
  - Training Accuracy: ~82%
  - Validation Accuracy: ~70%
  - Showed better learning but still suffered from overfitting (increasing validation loss).

#### Custom CNN Model

- **Design Principles**:
  - 8 convolutional layers (3x3 kernels), each followed by batch normalization, ReLU activation, and max pooling.
  - Interleaving pooling and batch normalization layers to improve feature extraction and regularization.
  - No dropout layers, to keep the architecture less complex than VGG16 but more expressive than AlexNet.
  - 3 fully connected layers for final classification.
  - Total Parameters: ~44.5 million.
- **Benefits**: Balances complexity and generalizability, specifically tuned via experimentation for the dataset’s size and diversity.

### Training Procedure

- **Framework**: PyTorch (see `CNN_vFinal.ipynb` for details)
- **Optimizers**: SGD with learning rate 0.002 (or 0.005 for some trials)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 4
- **Epochs**: Up to 100, with early stopping (patience=5, min_delta=0.05)
- **Hardware**: Supports GPU/CPU/MPS backends via auto-detection.

Training and validation progress is visualized with loss and accuracy curves for all models.

---

## Results & Evaluation

- **Custom CNN Model**:
  - Validation Accuracy: **94%**
  - Recall for Counterfeit Class: **86%**
  - Outperformed both AlexNet and VGG16 on this task.
- **AlexNet**: Capped at ~58.8% accuracy, unable to learn relevant visual features.
- **VGG16**: Achieved high training accuracy (82%) but lower validation accuracy (70%) and clear signs of overfitting.
- **Visualization**: See loss/accuracy plots in the notebook for learning dynamics and overfitting analysis.
- **Interpretation**: The custom model’s architecture effectively balances expressive power and regularization, making it suitable for small, imbalanced datasets.

---

## Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/varunmedi/counterfeit-ic-detection.git
    cd counterfeit-ic-detection
    ```

2. **Dependencies**
    - Python 3.x
    - PyTorch
    - torchvision
    - OpenCV
    - matplotlib
    - scikit-learn

    Install via pip:
    ```bash
    pip install torch torchvision opencv-python matplotlib scikit-learn
    ```

3. **Running the Notebook**
    - Open `CNN_vFinal.ipynb` in Jupyter or Google Colab.
    - Update dataset paths as needed.
    - Execute all cells to reproduce results and visualizations.

---

## Future Work

- Broaden data augmentation with brightness, contrast, and noise adjustments.
- Explore additional architectures (e.g., ResNet, EfficientNet).
- Experiment with ensemble methods and advanced regularization.
- Gather a larger, more diverse dataset for improved generalization.
- Integrate explainability/interpretability modules to help users understand what features the model uses for detection.

---

## References

- [PyTorch Documentation](https://pytorch.org/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- Academic literature on counterfeit detection and deep learning for industrial inspection.

---

**For detailed code, experiments, and results, see [`CNN_vFinal.ipynb`](CNN_vFinal.ipynb).**
