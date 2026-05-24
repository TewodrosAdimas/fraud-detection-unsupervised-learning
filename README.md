
---

# 📘 Fraud Detection Pipeline — Project Summary

This repository implements a complete **unsupervised fraud‑detection pipeline** combining classical clustering methods, anomaly‑detection algorithms, and a probabilistic **Bayesian Network** for behavioral modeling.

---

## 🔧 1. Data Processing & Feature Engineering

- Loaded and cleaned raw bank‑transaction data  
- Handled missing values, categorical encoding, and timestamp processing  
- Engineered key behavioral features:
  - `AmountBalanceRatio`
  - `TransactionCount`
  - `AccountID_freq`
  - One‑hot encoded occupation and channel variables  
- Discretized numeric features using **quantile‑based KBinsDiscretizer** for Bayesian Network learning

---

## 📊 2. Clustering Analysis

Applied **three clustering techniques** from the course:

- **K‑Means**
- **Hierarchical Clustering**
- **DBSCAN**

Evaluated clusters using:

- Silhouette Score  
- Davies–Bouldin Index  
- Cluster compactness & separation  
- Sensitivity to scaling and initialization  

These clusters were later used to support anomaly detection and behavioral segmentation.

---

## 🚨 3. Anomaly Detection

Implemented **three anomaly‑detection paradigms**:

### **A. Proximity‑based**
- k‑NN distance  
- Local Outlier Factor (LOF)

### **B. Cluster‑based**
- Distance to cluster centroids  
- Small‑cluster membership


Generated a final **binary anomaly label** (`anomaly = 0/1`) used for downstream Bayesian inference.

---

## 🧠 4. Bayesian Network Modeling

Performed full Bayesian Network learning:

### **Structure Learning**
- Learned DAG using score‑based search (BIC)
- Resulting structure captured:
  - CustomerAge → Occupation chain  
  - AccountBalance → TransactionAmount  
  - TransactionType → Channel routing  
  - AccountID_freq → TransactionCount  

### **CPD Learning**
- Estimated Conditional Probability Distributions using Bayesian estimators  
- Produced interpretable probabilistic relationships across:
  - Demographics  
  - Financial behavior  
  - Transaction patterns  
  - Account activity  

### **Inference**
- Computed conditional probabilities such as:  
  - P(AccountBalance | anomaly)  
  - P(Channel | TransactionType)  
  - P(Occupation | Age, Balance)

The Bayesian Network acts as a **behavioral reasoning engine**, complementing clustering‑based anomaly detection.

---

## 🚀 5. What This Pipeline Achieves

- Multi‑layer behavioral modeling  
- Unsupervised fraud pattern discovery  
- Probabilistic reasoning over customer and transaction features  
- Explainable anomaly detection  
- A foundation for hybrid fraud‑detection systems

---
