# Hybrid Intrusion Detection System

A research-oriented Hybrid Intrusion Detection System (IDS) that combines machine learning and rule-based analysis to detect malicious network traffic. The project employs a weighted ensemble of Random Forest, Logistic Regression, and an adaptive rule engine to improve detection performance while providing confidence estimation and model explainability.

---

## Overview

Intrusion Detection Systems (IDS) are essential for identifying malicious activities in computer networks. Traditional machine learning approaches often rely on a single classifier, which may struggle with complex attack patterns or produce high false alarm rates.

This project investigates a hybrid detection framework that combines the strengths of multiple machine learning models with rule-based analysis. Rather than relying on a single prediction, the system aggregates outputs from different components using a weighted ensemble strategy and estimates the confidence of each prediction.

The long-term objective is to develop an explainable and extensible IDS that can be evaluated on modern intrusion detection datasets and deployed for real-time network monitoring.

---

## Features

- Binary network intrusion detection (Normal / Attack)
- Random Forest classifier
- Logistic Regression classifier
- Adaptive rule-based detection
- Weighted hybrid ensemble
- Confidence score estimation
- Feature importance analysis
- SHAP-based model explainability
- Confusion matrix visualization
- Error analysis framework

---

## System Architecture

```text
                    Network Traffic
                           │
                           ▼
                  Data Preprocessing
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Random Forest   Logistic Regression   Rule Engine
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 Weighted Ensemble Decision
                           ▼
                 Confidence Score Estimation
                           ▼
                     Final Classification
                           ▼
                  Explainability (SHAP)
```

---

## Methodology

The current implementation consists of the following stages:

1. **Data Preprocessing**
   - Binary attack labeling
   - Label encoding of categorical features
   - Missing value handling
   - Feature scaling for Logistic Regression

2. **Machine Learning Models**
   - Random Forest
   - Logistic Regression

3. **Adaptive Rule Engine**
   - Rule thresholds automatically derived from the training dataset
   - Used as an additional decision source

4. **Hybrid Decision Fusion**
   - Weighted combination of model probabilities and rule-based output
   - Confidence estimation for every prediction

5. **Model Explainability**
   - Global feature importance using SHAP
   - Local prediction explanations using SHAP waterfall plots

---

## Dataset

This project currently uses the **NSL-KDD** dataset for binary intrusion detection.

### Required Files

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

Clone the repository:

```bash
git clone https://github.com/tamaghnakundu/network-intrusion-detection.git
cd network-intrusion-detection
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib shap
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP

---

## Future Work

- Comprehensive false positive and false negative analysis
- Cross-dataset evaluation using CICIDS2017 and UNSW-NB15
- Real-time packet capture and intrusion detection
- Deep learning-based detection models
- Explainable AI dashboard for security analysts
- Web-based visualization interface

---

## License

This project is intended for educational and research purposes.