# Medical Expense Prediction

A machine learning project for predicting medical expenses based on patient-related features such as age, gender, BMI, number of children, region, and smoking status.

The project covers the complete machine learning workflow, from data preparation and exploratory data analysis to model training, evaluation, hyperparameter tuning, and deployment using Streamlit.

## Project Overview

Medical expenses can vary significantly depending on several factors such as age, BMI, smoking status, and other demographic characteristics.

This project aims to build a regression-based machine learning system that learns from historical medical insurance data and predicts the estimated medical expenses for a given patient.

The project also provides an interactive Streamlit web application where users can enter patient information and obtain a predicted medical expense.

## Problem Statement

To develop a machine learning model capable of predicting medical expenses using demographic and health-related features.

The project focuses on:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature preprocessing
- Regression model development
- Model comparison
- Hyperparameter tuning
- Error and residual analysis
- Model deployment

## Objectives

- Clean and prepare a real-world-style medical insurance dataset.
- Perform exploratory data analysis to understand relationships between features.
- Handle missing values, duplicate records, inconsistent categorical values, and outliers.
- Prepare numerical and categorical features for machine learning.
- Train and compare multiple regression models.
- Tune the best-performing model using GridSearchCV.
- Evaluate models using appropriate regression metrics.
- Deploy the final model through an interactive Streamlit application.

## Dataset

The dataset contains the following features:

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `gender` | Gender of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `region` | Residential region |
| `smoker` | Smoking status |
| `expenses` | Medical expenses (target variable) |

The project intentionally includes data-quality issues such as missing values, duplicate records, inconsistent capitalization and spacing, and selected artificial outliers in order to demonstrate the complete data preprocessing workflow.

## Data Preparation & EDA

The data preparation stage included:

- Dataset inspection
- Statistical summary
- Missing-value analysis
- Duplicate detection
- Categorical-value analysis
- Distribution analysis
- Correlation analysis
- Outlier detection
- Data cleaning

Several visualizations were created during EDA, including:

- Age distribution
- BMI distribution
- Medical expense distribution
- Gender distribution
- Region distribution
- Smoking status distribution
- BMI vs. medical expenses
- Age vs. medical expenses
- Smoking status vs. medical expenses
- Correlation heatmap

## Data Preprocessing

The following preprocessing steps were performed:

1. Missing values were handled using appropriate strategies.
2. Duplicate records were removed.
3. Categorical values were standardized.
4. Artificial BMI and expense outliers were handled.
5. The target variable `expenses` was separated from the input features.
6. The dataset was divided into training and testing sets.
7. Numerical and categorical features were processed using a `ColumnTransformer`.
8. Categorical variables were encoded using `OneHotEncoder`.

The final cleaned dataset contained:

- 1326 records
- 7 columns

The train-test split used 80% training data and 20% testing data.

## Models Used

Five regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Tuned Gradient Boosting Regressor

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Percentage of predictions within ±10% of the actual expense

### Final Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4412.09 | 6398.14 | 0.6880 |
| Decision Tree | 3667.59 | 7270.49 | 0.5971 |
| Random Forest | 2844.81 | 5276.29 | 0.7878 |
| Gradient Boosting | 2795.42 | 5158.73 | 0.7972 |
| Tuned Gradient Boosting | 2785.11 | 5134.99 | 0.7990 |

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the Gradient Boosting model.

The parameters explored included:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

The best configuration was:

```text
n_estimators = 200
learning_rate = 0.05
max_depth = 2
min_samples_split = 2
min_samples_leaf = 1
```
## Final Model

The Tuned Gradient Boosting Regressor was selected as the final model.

It achieved:

- MAE: ₹2,785.11
- RMSE: ₹5,134.99
- R²: 0.7990

Although the Decision Tree achieved the highest percentage of predictions within ±10% error, Tuned Gradient Boosting provided the best overall performance based on MAE, RMSE, and R².

## Model Analysis

The project also includes:

- Actual vs. Predicted plots
- Residual plots
- Model comparison charts
- Error analysis
- Performance comparison across all models

These visualizations help evaluate how closely the predicted medical expenses match the actual expenses.

## Streamlit Application

The trained model was integrated into a Streamlit web application.

The application provides two sections:

### Prediction

Users can enter:

- Age
- Gender
- BMI
- Number of children
- Region
- Smoking status

The application then predicts the estimated medical expense using the tuned Gradient Boosting model.

### Model Evaluation

The application displays:

- Model performance comparison
- MAE, RMSE and R² scores
- ±10% prediction metric
- Actual vs. Predicted plots
- Residual plots
- Final model results
- Selected best model

## Project Structure

```text
medical-expense-prediction/
│
├── data/
│   ├── processed/
│   │   └── medical_insurance_cleaned.csv
│   │
│   └── raw/
│       ├── medical_insurance_messy.csv
│       ├── medical_insurance_with_smoker.csv
│       └── medical_insurance.csv
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── gradient_boosting_baseline_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── linear_regression_model.pkl
│   ├── preprocessor.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   ├── app.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```
## Project Links

- GitHub Repository: https://github.com/ddyano/medical-expense-prediction
- Live Streamlit App: https://medical-expense-prediction-3zrj2xwwsugjq2cvartbxy.streamlit.app/

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Plotly
- Streamlit
- Jupyter Notebook
- Git & GitHub

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/ddyano/medical-expense-prediction.git
