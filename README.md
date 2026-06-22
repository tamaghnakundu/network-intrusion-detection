# Hybrid Intrusion Detection System

A machine learning and rule-based hybrid intrusion detection framework using the NSL-KDD dataset.

---

## Overview

This project implements a Hybrid Intrusion Detection System (IDS) for detecting malicious network traffic. The framework combines multiple machine learning models with a rule-based detection engine and confidence estimation to improve reliability and reduce false alarms.

The project is being developed toward a research-oriented IDS framework with explainable AI and support for modern intrusion detection datasets.

---

## Features

### Data Preprocessing

- Binary attack classification (Normal / Attack)
- Label encoding of categorical features
- Missing value handling
- Feature scaling for Logistic Regression

### Machine Learning Models

- Random Forest Classifier
- Logistic Regression Classifier

### Hybrid Detection Framework

- Adaptive rule-based detector
- Weighted ensemble fusion
- Confidence score estimation

### Visualization

- Feature importance analysis
- Confusion matrix visualization
- Comparative model evaluation

---

## Architecture

```text
                    Network Traffic
                           │
                           ▼
                    Data Preprocessing
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 Random Forest      Logistic Regression    Rule Engine
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                Weighted Hybrid Ensemble
                           ▼
                  Confidence Estimation
                           ▼
                    Final Prediction
```

---

## Models Used

### Random Forest

- 500 estimators
- Balanced class weights
- Feature importance analysis

### Logistic Regression

- Standardized features
- Balanced class weights
- Probability estimation

### Adaptive Rule Engine

Thresholds are automatically computed from the training set using the 95th percentile of selected features:

- `count`
- `src_bytes`
- `srv_count`
- `dst_host_srv_count`

### Weighted Hybrid Ensemble

The final hybrid score is computed as:

```text
Hybrid Score =
0.6 × Random Forest Probability
+ 0.3 × Logistic Regression Probability
+ 0.1 × Rule Engine Prediction
```

Attack traffic is predicted when:

```text
Hybrid Score > 0.5
```

---

## Dataset

This project uses the NSL-KDD dataset.

Required files (download separately):

- `KDDTrain+.txt`
- `KDDTest+.txt`
- `feature_names.txt`

---

## Current Performance

| Metric | Value |
|----------|--------|
| Accuracy | 85% |
| Attack Precision | 97% |
| Attack Recall | 72% |
| F1-score | 83% |

### Confusion Matrix

|                | Predicted Normal | Predicted Attack |
|----------------|----------------:|----------------:|
| Actual Normal  | 9437 | 274 |
| Actual Attack  | 3552 | 9281 |

---

## Experiments Conducted

- Binary attack classification
- Random Forest hyperparameter tuning
- Logistic Regression with feature scaling
- Feature importance analysis
- Confusion matrix visualization
- Adaptive rule-based detection
- Weighted ensemble fusion
- Confidence score estimation

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Future Work

### Explainable AI

- SHAP-based feature attribution
- Local prediction explanations

### Error Analysis

- False positive analysis
- Misclassification analysis

### Modern Datasets

- CICIDS2017 support
- UNSW-NB15 support

### Real-Time IDS

- Live packet capture
- Real-time alerting
- Dashboard visualization

### Deep Learning Extensions

- Autoencoder-based anomaly detection
- PyTorch implementation
- Transformer-based intrusion detection

---

## Objective

The objective of this project is to develop a research-oriented hybrid intrusion detection framework that combines machine learning and rule-based analysis while providing confidence estimation and future explainability capabilities.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Implemented Components

- Random Forest classifier
- Logistic Regression classifier
- Feature scaling
- Feature importance analysis
- Confusion matrix visualization
- Adaptive rule engine
- Weighted hybrid ensemble
- Confidence score estimation

## Planned Extensions

- SHAP explainability
- False positive analysis
- CICIDS2017 support
- Real-time packet capture
- Deep learning models