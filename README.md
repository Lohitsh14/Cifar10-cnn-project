# CIFAR-10 Image Classification using Custom CNN

This project implements an image classification model using a Convolutional Neural Network (CNN) built from scratch (no pre-trained models).  
The model is trained and evaluated on the CIFAR-10 dataset (60,000 color images across 10 classes).

---

## Project Overview

- Build a CNN from scratch using TensorFlow/Keras  
- Train the model on CIFAR-10  
- Evaluate accuracy, precision, recall, F1-score  
- Visualize correct & incorrect predictions  
- Plot training and validation curves  

---

## Repository Contents

```text
cifar10-cnn-project/
│
├── cifar10_cnn.ipynb     # Main Jupyter notebook
└── README.md             # Project documentation
```

---

## Model Architecture

- Conv2D (32 filters) + ReLU + MaxPool  
- Conv2D (64 filters) + ReLU + MaxPool  
- Flatten  
- Dense (64 units) + ReLU  
- Dense (10 units) + Softmax  

---

## Training Results (3 Epochs)

- Training Accuracy: 65.76%  
- Validation Accuracy: 64.57%  
- Test Accuracy: 63.89%  
- Test Loss: 1.0374  

---

## Classification Metrics

Macro F1-score: **~0.64**

Strong classes:
- Automobile  
- Horse  
- Truck  

Moderate classes:
- Bird  
- Cat  
- Deer  

---

## Visualization

Notebook includes:
- Training vs validation accuracy plot  
- Training vs validation loss plot  
- Correct predictions  
- Incorrect predictions  

---

## How to Run

### Requirements:
- Python 3.9+  
- TensorFlow 2.x  
- NumPy  
- Matplotlib  
- scikit-learn  

### Run in Jupyter:
```bash
jupyter notebook
```
Open **cifar10_cnn.ipynb** and run all cells.

---

## Future Improvements

- Data Augmentation  
- Batch Normalization  
- More CNN layers  
- More training epochs  
- L2 regularization  
