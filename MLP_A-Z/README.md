# MLP Alphabet Recognition Guide

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Workflow](#workflow)
4. [How to Run](#how-to-run)
5. [Expected Results](#expected-results)

---

## Overview

This project implements a **multi-class MLP for handwritten alphabet recognition (A-Z)** using flattened 32x32 grayscale images. Each input image is classified into one of the 26 uppercase letters.

### Main characteristics
- **Task:** 26-class handwritten character recognition
- **Input format:** Flattened 32x32 grayscale images (`1024` features)
- **Model family:** Fully connected neural network (MLP)
- **Training strategy:** Standardization + safe augmentation + mixup regularization
- **Evaluation:** Overall accuracy, loss, and per-class precision/recall

---

## Requirements

### Python environment
- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Matplotlib

### Dataset
The notebook expects the alphabet image dataset to be available in the dataset path configured inside the preprocessing cells. Each image is resized to 32x32, normalized, and then flattened before training.

---

## Workflow

1. Load the raw alphabet images from disk
2. Resize each image to 32x32 grayscale
3. Encode labels as one-hot vectors
4. Split the data using stratification so every class remains represented
5. Apply safe augmentation to increase the diversity of the training set
6. Standardize the features before training
7. Apply mixup to create soft synthetic training examples
8. Train the MLP with regularization and learning-rate scheduling
9. Evaluate the model on the test split
10. Inspect per-class performance to identify difficult letters

---

## How to Run

1. Open `INF3721_TP3_Groupe_01.ipynb` in Jupyter or VS Code
2. Run the data loading and preprocessing cells first
3. Run the stratified split cell
4. Run the safe augmentation cell to create the augmented training set
5. Run the MLP v6 training cell
6. Run the visualization cell to inspect training curves
7. Run the per-class analysis cell to see which letters are hardest to classify

### Notes
- The training cell automatically uses the augmented dataset if it already exists
- If the augmentation cell was not run manually, the training cell can regenerate it automatically
- The notebook is designed to be robust to cell order mistakes

---

## Expected Results

When the pipeline is executed correctly, you should see:
- A balanced train/test split with all 26 classes represented
- An augmented training set larger than the original raw split
- A training process that gradually improves validation performance
- Final test accuracy and loss printed after evaluation
- A per-class report that highlights letters with lower recall, such as visually similar characters

---

## Example Usage

### Example: Train the MLP
```python
# After preprocessing and augmentation
history = model.fit(
    X_train_final,
    y_train_final,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

### Example: Evaluate the model
```python
loss, acc = model.evaluate(X_test_mlp, y_test, batch_size=32, verbose=1)
print(f"Test accuracy: {acc * 100:.2f}%")
print(f"Test loss: {loss:.4f}")
```
