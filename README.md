
# Credit Card Fraud Detection System (Batch + Streaming)

This project implements a **Credit Card Fraud Detection System** using **PySpark**, **HDFS**, and **Kafka**. It supports both **batch prediction** and **real-time streaming prediction**.

The system uses machine learning models (Logistic Regression / Random Forest) trained on the **Credit Card Fraud dataset** and demonstrates how to integrate batch and streaming pipelines in Spark.

**Dataset is derived from kaggle credit card dataset**
---

## Table of Contents
1. [Features](#features)   
2. [Installation / Setup](#installation--setup)  
3. [Project Notes](#project-notes)  

---

## Features

- Load credit card transactions from HDFS  
- Preprocess data using PySpark  
- Train **Logistic Regression** or **Random Forest** models  
- Evaluate models with AUC, Precision, Recall, F1-Score  
- Stream live transactions from Kafka and predict fraud in real-time  
- View batch and streaming predictions in console  

---
## Installation and Setup
### 1. Create virtual environment and install dependencies
### 2. Install HDFS,Spark,Kafka
### 3. pip install pyspark pandas findspark in VS Code
### 4. Run Spark, Kafka Zookeeper
---
## Project Notes

Dataset is highly imbalanced (fraud cases are rare). Metrics like Recall and F1-score are more important than raw accuracy.

Both Logistic Regression and Random Forest are used. You can switch between them by loading the respective saved models.(Random Forest has better accuracy)

Streaming uses Kafka and Spark Structured Streaming; batch uses standard Spark DataFrame ML pipeline.