# 🧠 MNIST Digit Classification using CNN

This project implements an end-to-end deep learning pipeline for handwritten digit classification using the MNIST dataset. The objective is to build and evaluate neural network models capable of accurately classifying grayscale images of digits (0–9).

---

## 📌 Project Overview

The MNIST dataset consists of 28×28 grayscale images representing handwritten digits. This project follows a structured machine learning workflow:

1. Data ingestion and validation  
2. Exploratory Data Analysis (EDA)  
3. Data preprocessing  
4. Baseline model benchmarking  
5. Convolutional Neural Network (CNN) training  
6. Model evaluation  

---

## 📊 Dataset

- Source: MNIST (Kaggle CSV format)
- Training samples: ~42,000
- Image size: 28 × 28 pixels
- Classes: 10 digits (0–9)
- Format: 784 pixel features + label column

---

## 🔍 Data Preprocessing

The preprocessing pipeline includes:

- Separation of features and labels
- Conversion to `float32`
- Pixel normalization (scaling values to range 0–1)
- Reshaping to `(28, 28, 1)` for CNN input
- Stratified train-validation split

---

### 🔹 Baseline Model (MLP)

A simple fully connected neural network was implemented to establish a benchmark performance.

**Architecture:**
- Flatten layer
- Dense layer (ReLU activation)
- Output layer (Softmax activation)

**Baseline accuracy:** ~96–98%

---

### 🔹 Final Model (CNN)

A Convolutional Neural Network was built to capture spatial features in the digit images.

**Architecture includes:**
- Convolutional layers
- Batch Normalization
- MaxPooling layers
- Dropout for regularization
- Fully connected dense layers
- Softmax output layer

**Loss Function:** sparse_categorical_crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

The CNN achieves validation accuracy of approximately **99%+**, significantly outperforming the baseline model.

---

## 📈 Evaluation

Model performance is evaluated using:

- Training & validation accuracy
- Training & validation loss
- Confusion matrix
- Classification metrics (Precision, Recall, F1-score)

The CNN demonstrates strong generalization and stable convergence.

---

## 🛠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 🎯 Key Takeaways

- CNNs significantly improve performance over basic MLP models for image data.
- Proper normalization and structured preprocessing are critical for stable training.
- Even relatively simple CNN architectures can achieve high accuracy on structured datasets like MNIST.

---

⭐ If you found this project useful, feel free to fork or star the repository.
