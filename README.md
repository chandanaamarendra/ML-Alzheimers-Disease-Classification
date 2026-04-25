# 🧠 Alzheimer's Disease Classification Using Machine Learning

## Project Overview

This project focuses on the early detection of Alzheimer's disease using Machine Learning models based on cognitive and behavioral assessments.

The main objective is to help healthcare professionals identify Alzheimer's risk at an early stage and support medical diagnosis using data-driven insights.

This project was completed as part of my Machine Learning Internship in the Healthcare Domain from AI Variant.

---

## Problem Statement

Alzheimer's disease is a progressive neurological disorder that affects memory, thinking ability, and behavior.

Early diagnosis is extremely important for better treatment planning, improved patient care, and medical decision-making.

This project predicts Alzheimer's disease risk using patient assessment data and machine learning techniques.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Machine Learning

---

## Dataset Details

The dataset contains patient information including:

- Age
- Gender
- Education Level
- Cognitive Test Scores
- Medical History
- Lifestyle Factors

### Target Variable

- 0 → No Alzheimer's
- 1 → Alzheimer's Disease

---

## Data Preparation

### Data Cleaning

Performed the following steps:

- Removed irrelevant columns
- Checked for duplicate records
- Verified missing values
- Prepared clean data for modeling

### Feature Selection

Top selected features using SelectKBest:

- MMSE
- Functional Assessment
- Memory Complaints
- Behavioral Problems
- ADL Score

These features were identified as the strongest indicators for Alzheimer's prediction.

---

## Models Compared

The following machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

<img width="1181" height="514" alt="image" src="https://github.com/user-attachments/assets/3e5500fc-4c73-473e-946b-d58b7667113b" />

---

## Best Model Performance

## Random Forest Classifier

### Performance Metrics

- Accuracy: 95.58%
- Precision: 96.52%
- Recall: 90.84%
- F1 Score: 93.60%

Random Forest achieved the best overall performance and was selected as the final model for deployment.

---

## Streamlit Web Application

A user-friendly Streamlit web application was developed where healthcare professionals can input patient details and predict Alzheimer's disease risk instantly.

### Features of the App

- Risk prediction (Low / Moderate / High)
- Probability score display
- Clinical interpretation
- Feature importance visualization

<img width="1230" height="572" alt="image" src="https://github.com/user-attachments/assets/98ffb400-0c30-46bd-9d79-7b1e18395aa3" />

---

## Project Outcome

This project helped strengthen my practical skills in:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Selection
- Model Building
- Model Evaluation
- Model Deployment using Streamlit
- Healthcare Analytics

---

## Author

### A. Chandana

Aspiring Data Analyst | Data Scientist  
SQL | Python | Power BI | Machine Learning | Healthcare Analytics
