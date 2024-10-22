# Credit-Risk-Modeling


This project focuses on building a robust Credit Risk Prediction model, identifying individuals based on different approval flags (`P1`, `P2`, `P3`, `P4`). The project involves extensive data preprocessing, model fitting, and hyperparameter tuning using **Random Forest**, **XGBoost**, and **Decision Trees**. The best results were obtained using XGBoost, fine-tuned with grid search and hyperparameter optimization.

# I am still working on the deployment of this project, where I will give user an interface to upload 2 excel files & return their predictions, where I'll use Flask for Web App

## Project Overview
The main goal of this project is to predict credit risk using a classification model, employing several machine learning techniques. The steps include:

1. **Data Preprocessing Pipeline**
   - Data Cleaning
   - Chi-square tests for categorical features
   - Variance Inflation Factor (VIF) for multicollinearity removal
   - ANOVA tests for numerical features
   - Feature Encoding (OneHot & Ordinal)
   - Feature Scaling

2. **Machine Learning Models**
   - **Random Forest**
   - **XGBoost**
   - **Decision Tree**
     
3. **Best Accuracy received from XGBoost i.e `78%`** 

4. **Model Tuning**
   - Hyperparameter Tuning for XGBoost
   - Grid Search for optimized parameters
   - Model Evaluation using Accuracy, Precision, Recall, F1 Score

5. **Saving the Best Model**
   - Exporting the trained model using `joblib` for deployment.

## Data Preprocessing Pipeline

The data pipeline includes the following key steps:

- **Data Cleaning**: Removed rows with invalid values (`-99999`) and columns with excessive missing data.
- **Chi-Square Test**: Applied to categorical columns (`MARITALSTATUS`, `EDUCATION`, `GENDER`, etc.) to assess their relevance to the target variable.
- **VIF (Variance Inflation Factor)**: Calculated to eliminate features with multicollinearity issues.
- **ANOVA Test**: Used for numerical columns to identify significant features.
- **Feature Encoding**: Applied OneHot encoding for categorical variables and ordinal encoding for educational levels.
- **Scaling**: Standardized numerical features that significantly impacted the model's performance.

```python
# Sample code for preprocessing
df_encoded = data_preprocessing_pipeline(a1_path, a2_path)
