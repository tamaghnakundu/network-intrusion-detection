# Hybrid Intrusion Detection System

A hybrid intrusion detection framework that combines machine learning and rule-based analysis for network attack detection using the NSL-KDD dataset.

## Overview

This project explores the use of ensemble machine learning and rule-based detection for identifying malicious network traffic. The current implementation combines Random Forest, Logistic Regression, and an adaptive rule engine through a weighted decision fusion strategy.

Future work focuses on explainable AI, modern intrusion detection datasets, and real-time traffic analysis.

---

## Features

- Binary attack classification
- Random Forest classifier
- Logistic Regression classifier
- Adaptive rule-based detection
- Weighted hybrid ensemble
- Confidence score estimation
- Feature importance analysis
- Confusion matrix visualization

---

## Architecture

```text
Network Traffic
       │
       ▼
Data Preprocessing
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
RF     LR   Rule Engine
 \      |      /
  \     |     /
   Weighted Ensemble
          │
          ▼
 Confidence Score
          │
          ▼
 Final Prediction
```

---

## Dataset

- NSL-KDD
- Binary classification (Normal / Attack)

Required files:

- `KDDTrain+.txt`
- `KDDTest+.txt`
- `feature_names.txt`

---

## Current Results

| Metric | Value |
|---------|------:|
| Accuracy | 85% |
| Precision | 97% |
| Recall | 72% |
| F1-score | 83% |

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Future Work

- SHAP explainability
- False positive analysis
- CICIDS2017 support
- Real-time intrusion detection
- Deep learning models

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib