# Hybrid Neural Network + SVM Classifier

A hybrid deep learning model that combines Neural Networks (NN), Random Fourier Features (RFF), and Support Vector Machines (SVM) for robust multi-class classification on high-dimensional datasets. Designed for scenarios where feature extraction with NN improves separability before classification with an SVM.

---

## 🚀 Project Overview

This project aims to:

- Combine **Neural Networks** for non-linear feature extraction
- Use **Random Fourier Features (RFF)** to approximate kernel mappings
- Train a **Support Vector Machine (SVM)** for the final classification stage
- Handle **imbalanced datasets** using techniques like SMOTE
- Support both binary and multi-class classification using One-vs-Rest

This architecture is particularly useful when raw features are not linearly separable, but transformed features extracted from NN layers provide better class discrimination.

---

## 📂 Project Structure

| File / Folder              | Description                                                                                      |
|---------------------------|--------------------------------------------------------------------------------------------------|
| `Hybrid_optimizer.py`      | Defines the hybrid model trainer: neural network, RFF, and SVM with separate optimizers         |
| `Hybrid_trainer.py`        | Contains the One-vs-Rest wrapper for multi-class classification using the hybrid model          |
| `NNTransformer.py`         | Contains custom neural network layers and transformations for feature extraction                |
| `RFFTransform.py`          | Implements the Random Fourier Features (RFF) layer                                              |
| `main.py`                  | Entry point: loads data, preprocesses it, builds and trains the hybrid model                    |
| `utils.py`                 | Utility functions for data cleaning, normalization, and visualization                           |
| `requirements.txt`         | Lists Python dependencies to set up the environment                                             |
| `README.md`                | This documentation file                                                                          |

---

## 📦 Requirements

Before running the project, make sure you have:

- Python ≥ 3.8
- TensorFlow ≥ 2.x
- Scikit-learn
- imbalanced-learn
- NumPy
- Matplotlib
- Pandas
