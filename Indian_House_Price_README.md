Indian House Price Prediction

This project predicts house prices in India using machine learning. The dataset includes property details like location, size, amenities, and price. A Random Forest Regressor is used to model price predictions after data preprocessing and feature engineering.

Project Overview:
Goal: Predict property prices (in Lakhs) based on features such as location, size, property type, and amenities.
Dataset: india_housing_prices.csv
Algorithm: Random Forest Regressor (sklearn)

Steps in the Notebook:
Import Libraries
  Used pandas, numpy, matplotlib, seaborn, and sklearn for analysis, visualization, and modeling.
Load Dataset
  Reads the CSV file into a pandas DataFrame.
  Displays shape, column types, and first few rows for an overview.
Data Cleaning
  Removes rows with missing target (Price_in_Lakhs).
  Fills missing categorical values with 'Unknown' and numerical values with median.
Feature Selection
  Splits features (X) and target (y).
  Identifies categorical vs numerical columns.
Preprocessing Pipeline
  Uses ColumnTransformer to one-hot encode categorical columns.
  Passes numerical columns directly.
Model Training
Trains a Random Forest Regressor with parameters:
  n_estimators=30
  max_depth=20 (to prevent overly deep trees)
  max_features='sqrt'
  n_jobs=3 (uses 3 CPU cores)
  random_state=42
Model Evaluation
Metrics calculated on test data:
  MAE (Mean Absolute Error)
  RMSE (Root Mean Squared Error)
  R² Score (Coefficient of Determination)
  
Feature Importance Visualization
  Plots top 15 features affecting house price predictions.

Key Results
  MAE: ~99.36
  RMSE: ~115.64
  R²: ~0.33 (model explains ~33% of price variation)
  Most important feature: Price_per_SqFt, followed by Size_in_SqFt.

How to Run
  Clone or download this repository.
  Install dependencies:
  pip install pandas numpy matplotlib seaborn scikit-learn
  Open the notebook in Jupyter:
  jupyter notebook Indian_House_Price.ipynb
  Run all cells sequentially.

Future Improvements
  Try hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
  Compare with other models like Gradient Boosting or XGBoost.
  Add geospatial analysis for location-based insights.
  Collect more recent or high-quality data to improve accuracy.
