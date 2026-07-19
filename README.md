# Hybrid Network Intrusion Detection System

A modular, explainable Hybrid Network Intrusion Detection System (IDS) built using machine learning and rule-based detection. The system combines Random Forest, Logistic Regression, and an adaptive Rule Engine through a weighted ensemble to improve intrusion detection performance while providing interpretable predictions using SHAP.

---

## Features

- Modular Python project architecture
- Random Forest classifier
- Logistic Regression classifier
- Adaptive Rule-Based anomaly detector
- Weighted Hybrid Ensemble
- Confidence score estimation
- SHAP-based model explainability
- Automated evaluation and visualization pipeline
- Reproducible experiments

---

## Project Structure

```text
network-intrusion-detection/
│
├── data/
│   ├── feature_names.txt
│   └── README.md
│
├── notebooks/
│   └── hybrid_ids_analysis.ipynb
│
├── results/
│   ├── figures/
│   ├── reports/
│   └── tables/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   └── visualization/
│
├── outputs/              # Generated automatically (ignored by Git)
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset

This project uses the **NSL-KDD** benchmark dataset for network intrusion detection.

The dataset is **not included** in this repository.

Download the following files and place them inside the `data/` directory:

```
KDDTrain+.txt
KDDTest+.txt
```

The repository already includes:

```
feature_names.txt
```

---

# Methodology

The proposed IDS consists of four major components.

## 1. Data Preprocessing

- Load NSL-KDD dataset
- Label encoding of categorical features
- Missing value handling
- Feature preparation
- Binary attack classification

---

## 2. Machine Learning Models

### Random Forest

- Ensemble tree classifier
- High detection capability
- Feature importance analysis

### Logistic Regression

- Linear classifier
- Standardized features
- Probability calibration

---

## 3. Rule-Based Detector

A lightweight adaptive detector that identifies suspicious traffic using feature-based heuristics.

The rule engine complements the machine learning models by improving robustness on uncertain samples.

---

## 4. Hybrid Ensemble

The final prediction is obtained using a weighted ensemble:

| Component | Weight |
|-----------|--------|
| Random Forest | 0.60 |
| Logistic Regression | 0.30 |
| Rule Engine | 0.10 |

The ensemble also computes a confidence score for every prediction.

---

# Explainability

The project integrates **SHAP (SHapley Additive Explanations)** to interpret model decisions.

Available visualizations include:

- SHAP Summary Plot
- SHAP Beeswarm Plot
- SHAP Waterfall Plot
- Random Forest Feature Importance

---

# Results

Representative outputs are provided in the `results/` directory.

## Included

### Figures

- Hybrid IDS Confusion Matrix
- Feature Importance
- SHAP Summary
- SHAP Beeswarm

### Reports

- Random Forest Classification Report
- Logistic Regression Classification Report
- Hybrid IDS Classification Report

### Tables

- Feature Importance CSV
- Sample Predictions

---

# Installation

Clone the repository

```bash
git clone https://github.com/tamaghnakundu/network-intrusion-detection.git

cd network-intrusion-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

After downloading the NSL-KDD dataset into the `data/` folder:

```bash
python main.py
```

The pipeline automatically:

- loads the dataset
- preprocesses the data
- trains all models
- evaluates performance
- generates explainability plots
- saves reports
- computes hybrid predictions

Generated files are written to the `outputs/` directory.

---

# Repository Contents

| Directory | Description |
|------------|-------------|
| `src/` | Source code |
| `data/` | Dataset configuration |
| `results/` | Curated figures and reports |
| `outputs/` | Auto-generated artifacts |
| `notebooks/` | Research notebook |

---

# Technologies Used

- Python
- Scikit-learn
- SHAP
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# Future Improvements

- Deep learning based IDS
- Streaming intrusion detection
- Real-time packet analysis
- AutoML for model selection
- Explainable ensemble optimization
- Web-based IDS dashboard

---

# Author

**Tamaghna Kundu**

B.Tech Computer Science and Engineering

SOA University

Interested in Machine Learning, Cybersecurity, and Explainable AI.