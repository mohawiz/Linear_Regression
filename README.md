Project Overview: Linear Regression on Advertising Budget and Sales Data
Data Loading:
Imported necessary libraries: import numpy as np, import pandas as pd, import seaborn as sns, import matplotlib.pyplot as plt.
Loaded the dataset from Kaggle, containing information on advertising budget and sales.
Exploratory Data Analysis (EDA):
Displayed the initial rows of the dataset to understand its structure.
Provided a summary description of the dataset using descriptive statistics.
Data Cleaning:
Identified and removed the first unnamed column that may have resulted from the index during data loading.
Correlation Analysis:
Computed the correlation coefficients between the 'Sales' column and all other columns.
Displayed a scatter plot (sns.scatterplot) to visually assess the correlation between independent variable 'TV Ads' and dependent variable 'Sales'.
Heatmap Visualization:
Plotted a heatmap (sns.heatmap) to visualize the correlation matrix, highlighting the relationships between different parameters.
Observed the highest correlation score of 'TV Ads' (0.78) with 'Sales', indicating a strong positive correlation.
Data Preparation:
Created two variables, X and y, where X contains all columns except 'Sales', and y contains only the 'Sales' column.
Standardization:
Imported the StandardScaler from scikit-learn (from sklearn.preprocessing import StandardScaler).
Standardized the feature variables using the StandardScaler technique to ensure consistent scales for modeling.
Model Training:
Trained a Linear Regression model (model.fit(X_train, y_train)) using the training data (X_train, y_train).
Model Prediction:
Utilized the trained model to predict the target variable on the testing set (y_pred = model.predict(X_test)).
Evaluation Metrics:
Calculated and printed key regression evaluation metrics:Mean Absolute Error (MAE): 1.1435743461463297
Mean Squared Error (MSE): 2.4290707066579964
Root Mean Squared Error (RMSE): 1.5585476273306493.
Conclusion:
Summarized the findings, highlighting the predictive performance of the Linear Regression model on the dataset.
Libraries Used:
numpy, pandas, seaborn, matplotlib.pyplot for data manipulation and visualization.
StandardScaler from sklearn.preprocessing for feature standardization.
accuracy_score, mean_absolute_error, mean_squared_error from sklearn.metrics for model evaluation.
