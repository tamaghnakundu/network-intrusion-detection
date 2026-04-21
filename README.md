# Network Intrusion Detection

Machine learning-based network intrusion detection using the NSL-KDD dataset.

## Overview

This project implements and compares multiple machine learning models for detecting malicious network traffic.

Models explored:

- Random Forest (baseline)
- Logistic Regression (scaled)

## Dataset

This project uses the NSL-KDD dataset.

Required files (download separately):

- KDDTrain+.txt
- KDDTest+.txt

## Features

- Binary attack classification
- Categorical feature encoding
- Feature scaling for Logistic Regression
- Random Forest feature importance analysis
- Confusion matrix visualization
- Comparative model evaluation

## Results

### Random Forest Baseline

- Accuracy: 77%
- Attack Recall: 62%

### Scaled Logistic Regression

- Accuracy: 84%
- Attack Recall: 79%

## Key Finding

Scaled Logistic Regression outperformed the Random Forest baseline on attack recall.

## Experiments Conducted

- Hyperparameter tuning for Random Forest
- Investigation of the `difficulty` feature
- Feature importance analysis
- Comparative model evaluation

## Future Work

- Live traffic detection
- Real-time alerting
- PyTorch-based anomaly detection

## Files

- ids.ipynb — notebook implementation
- ids.py — standalone Python script

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib
```