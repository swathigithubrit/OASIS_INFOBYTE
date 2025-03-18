# OASIS_INFOBYTE

1. Email Spam Classification
Overview
This project classifies emails as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and Machine Learning (ML) techniques. It processes text data, extracts features using TF-IDF, and trains models like Naïve Bayes to detect spam emails effectively.

Dataset
The dataset contains labeled email messages:
  Spam: Unwanted or malicious emails.
  Ham: Genuine, non-spam emails.

Features & Preprocessing
-> Data Cleaning – Removing special characters, numbers, and stopwords.
-> Tokenization & Stemming – Breaking text into words and converting them into root forms.
-> TF-IDF Vectorization – Converting text into numerical format for model training.
-> Label Encoding – Mapping spam as 1 and ham as 0.

Machine Learning Models Used
-> Naïve Bayes Classifier (Best for text classification)
-> Logistic Regression (For comparison)
-> Support Vector Machines (SVM) (For improved classification)


Model Evaluation
-> Accuracy Score – Measures overall model performance.
-> Precision & Recall – Ensures correct spam detection.
-> Confusion Matrix – Visualizes classification results.

Results & Insights
Achieved high spam detection accuracy using Naïve Bayes.
The confusion matrix helps analyze false positives and false negatives.
The model effectively reduces spam emails while minimizing misclassification of ham emails.

Conclusion:-
This project successfully demonstrates how NLP and Machine Learning can be used to classify emails as spam or not spam. The Naïve Bayes classifier, with TF-IDF feature extraction, provides a reliable spam detection system. The model achieves high accuracy, effectively reducing unwanted emails while minimizing false positives. Future improvements may include deep learning models and real-time email filtering for enhanced performance.


2. Sales Prediction
Overview
This project analyzes historical sales data to forecast future sales trends. By leveraging regression techniques, it identifies key factors influencing sales and provides predictive insights to support data-driven decision-making.

Dataset Information
The dataset used in this project contains information about advertising expenditures and corresponding sales figures. The key columns include:
-> TV – Budget spent on TV advertisements (in thousands of dollars).
-> Radio – Budget spent on Radio advertisements (in thousands of dollars).
-> Newspaper – Budget spent on Newspaper advertisements (in thousands of dollars).
-> Sales – Number of units sold (dependent variable).

Key Features
-> Performs Exploratory Data Analysis (EDA) to understand relationships between advertising channels and sales.
-> Uses Linear Regression to predict future sales based on input variables like TV, Radio, and Newspaper ads.
-> Evaluates model performance using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
-> Provides visualizations such as correlation heatmaps and scatter plots for better data interpretation.

Technologies Used
-> (pandas, NumPy, matplotlib, seaborn, scikit-learn)
-> Machine Learning (Linear Regression)

Conclusion
This project helps businesses understand sales patterns and optimize marketing strategies by predicting sales outcomes based on historical advertising expenditures.



3. Unemployment Analysis in India
Overview 
This project analyzes unemployment trends in India using Python. It integrates multiple datasets, performs data cleaning, and generates insightful visualizations to understand employment patterns, regional disparities, and unemployment trends over time.

Datasets Used
-> Unemployment in India.csv - Contains region-wise unemployment rates, employment estimates, and labor participation rates.
-> Unemployment_rate_upto_11_2020.csv - Includes additional attributes such as geographical coordinates (longitude, latitude) and zone classification.

Technologies Used
-> Python (Pandas, Matplotlib, Seaborn, Plotly)
-> Jupyter Notebook / VS Code
-> Data Visualization (Looker Studio, Power BI - optional for extended analysis)


Key Features
-> Data Cleaning and Preprocessing
-> Exploratory Data Analysis (EDA)
-> Trend Analysis using Line Plots
-> Regional Unemployment Rate Analysis using Box Plots
-> Correlation Heatmap to Identify Relationships between Features
-> Geospatial Analysis with Scatter Plots (using longitude & latitude)
-> Interactive Visualizations with Plotly (optional for deeper insights)

Visualizations
-> Unemployment Rate Trend Over Time - Identifies changes in unemployment across different time periods.
-> Unemployment Rate Distribution - Examines how unemployment rates vary within the dataset.
-> Unemployment Rate by Region - Highlights variations in unemployment across different Indian states.
-> Correlation Heatmap - Shows relationships between different employment-related factors.
-> Scatter Plot of Unemployment by Location - Provides a geographical view of unemployment trends.

Conclusion
The analysis provides a comprehensive view of unemployment trends in India. It highlights regions with the highest and lowest unemployment rates, helping policymakers focus on necessary interventions. Temporal trends reveal economic impacts, such as COVID-19 affecting employment patterns. Geospatial visualizations offer insights into employment trends across different zones, aiding in targeted decision-making.
