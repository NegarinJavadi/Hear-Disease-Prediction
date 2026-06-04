This project focuses on predicting the presence of heart disease using machine learning techniques.
The work is inspired by the research paper "Prediction of Heart Disease Using a Combination of Machine Learning and Deep Learning" (2021) and uses the UCI Heart Disease Cleveland dataset, one of the most widely used benchmark datasets in medical machine learning research. The dataset contains information from 303 patients and includes 14 clinical features such as age, blood pressure, cholesterol level, chest pain type, and maximum heart rate.

The goal is to build a binary classification model that predicts whether a patient is likely to have heart disease:
  0 = No Heart Disease
  1 = Heart Disease

Some of the available features include:
  Age
  Sex
  Chest Pain Type
  Resting Blood Pressure
  Cholesterol
  Fasting Blood Sugar
  Resting ECG Results
  Maximum Heart Rate
  Exercise-Induced Angina
  ST Depression
  Number of Major Vessels
  Thalassemia

Five machine learning algorithms were trained and evaluated:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Gradient Boosting

To obtain reliable and unbiased results, the models were evaluated using:
  5-Fold Cross Validation
  Accuracy
  Precision
  Recall
  F1 Score
  AUC-ROC (main evaluation metric)

AUC-ROC was selected as the primary metric because it measures how well a model can distinguish between patients with and without heart disease across different classification thresholds.

The experiments showed that ensemble-based methods achieved the strongest performance.
Test Accuracy: ~88%
AUC-ROC: > 0.90
These results indicate that the model can effectively identify patterns associated with heart disease and provide strong predictive performance on unseen data.

Paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8266441/

Conclusion:
This project demonstrates how classical machine learning algorithms can be applied to clinical patient data for heart disease prediction. By comparing multiple models under the same evaluation framework, it provides a practical benchmark for understanding the strengths and limitations of different classification approaches in healthcare applications.
