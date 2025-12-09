
# 💳 Credit Card Fraud Detection System (Batch + Streaming)

This project implements a robust **Credit Card Fraud Detection System** using **Apache PySpark**, **HDFS**, and **Apache Kafka**. It is engineered to handle both **batch prediction** and **real-time streaming prediction** workflows.

The system utilizes machine learning models (specifically **Logistic Regression** and **Random Forest**) trained on a representative fraud dataset to classify transactions as legitimate or fraudulent, demonstrating a practical integration of batch and streaming pipelines within the Spark ecosystem.

> **NOTE:** The dataset used is derived from the popular Kaggle Credit Card Fraud Detection dataset, focused on anonymized European card transactions.

---

## 🧭 Table of Contents
1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites and Setup](#prerequisites-and-setup)
4. [Execution Guide](#execution-guide)
5. [Project Notes and ML Details](#project-notes-and-ml-details)

---

## ✨ Key Features

This system provides comprehensive functionality across data handling, model training, and prediction:

* **Data Ingestion:** Loads historical credit card transaction data directly from **HDFS** for batch processing.
* **Data Preprocessing:** Uses **PySpark** DataFrames and ML utility functions for efficient data preparation and feature engineering.
* **Model Training:** Supports training and persistence of two distinct models:
    * **Logistic Regression (LR)**
    * **Random Forest (RF)** (Primary model due to superior performance)
* **Model Evaluation:** Provides thorough model assessment using critical metrics for imbalanced data, including **AUC**, **Precision**, **Recall**, and **F1-Score**.
* **Real-Time Prediction:** Integrates with **Kafka** to consume live transaction streams and uses **Spark Structured Streaming** for immediate, low-latency fraud prediction.
* **Output:** Displays both batch and streaming predictions in the console for monitoring.

---

## 🏗️ Architecture Overview

The system operates on two parallel tracks:

* **Batch Pipeline:** Historical data in HDFS $\rightarrow$ PySpark Processing $\rightarrow$ Model Training/Evaluation $\rightarrow$ Persisted Model.
* **Streaming Pipeline:** Live Transactions $\rightarrow$ Kafka Topic $\rightarrow$ Spark Structured Streaming $\rightarrow$ Loaded Model $\rightarrow$ Real-time Prediction.



---

## ⚙️ Prerequisites and Setup

This project requires a distributed computing environment and specific Python dependencies.

### 1. Environmental Dependencies

The following external services are mandatory and must be running:

* **Apache HDFS** (Hadoop Distributed File System)
* **Apache Spark** (Version 3.x is recommended)
* **Apache Kafka** and its dependency, **ZooKeeper**

### 2. Python Environment Setup

Create a virtual environment and install the necessary Python packages:

```bash
# Create a new virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install core dependencies
pip install pyspark pandas findspark
```
### 3. Execution Checklist

Before running the main scripts, ensure the following services are active:

1. **Start Kafka ZooKeeper service.**  
2. **Start Kafka Broker service.**  
3. **Ensure HDFS is running** (e.g., via `start-dfs.sh`).  
4. **Ensure Spark is accessible**, either locally or through a cluster manager.

---

## ▶️ Execution Guide

The project typically involves **three main phases**:

1. **Phase 1: Batch Model Training (One-time)**  
2. **Phase 2: Start the Kafka Producer**
3. **Phase 3: Start the Streaming Detector**
---
## 📝 Project Notes and ML Details 
### Data Characteristics
The dataset exhibits a severe class imbalance (fraud cases are very rare, typically $< 0.2\%$). This characteristic significantly impacts model development and evaluation.
Metric Focus: Due to imbalance, metrics such as Recall, F1-Score, and AUC-ROC are far more meaningful indicators of model performance than simple raw Accuracy.Recall is prioritized to minimize False Negatives (i.e., actual fraud cases that are missed).
### Model Selection
The system is configured to use either a Logistic Regression or a Random Forest model.
Performance: The Random Forest model consistently achieves better overall performance metrics for this dataset and is the recommended default for production.
Switching Models: The streaming script loads the model from a predefined HDFS path. You can switch models by updating the path to point to the saved LR or RF model file
