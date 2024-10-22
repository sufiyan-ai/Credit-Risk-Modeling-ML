# Credit-Risk-Modeling


This project focuses on building a robust Credit Risk Prediction model, identifying individuals based on different approval flags (`P1`, `P2`, `P3`, `P4`). The project involves extensive data preprocessing, model fitting, and hyperparameter tuning using **Random Forest**, **XGBoost**, and **Decision Trees**. The best results were obtained using XGBoost, fine-tuned with grid search and hyperparameter optimization.

## Project Overview
This **Credit Risk Modeling** project leverages machine learning to predict the probability of loan approval or default based on customer data, including financial and behavioral variables. By applying algorithms such as XGBoost, Random Forest, and Decision Trees, the model identifies patterns in historical data to provide accurate risk assessments. The solution supports financial institutions in making informed lending decisions, reducing the risk of defaults, and improving overall credit portfolio management.

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

## Hyperparameter Tuning for XGBoost

To optimize XGBoost, I performed a comprehensive grid search across the following parameters:

- **n_estimators**: [50, 100, 200]
- **max_depth**: [3, 5, 7]
- **learning_rate**: [0.01, 0.1, 0.2]

The final model was trained with the best parameters found via grid search.

```python
from sklearn.model_selection import GridSearchCV

# Define the XGBClassifier and parameter grid
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
grid_search.fit(x_train, y_train)

# Output the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
```
## I am still working on the deployment of this project, where I will give user an interface to upload 2 excel files & return their predictions, where I'll use Flask for Web App

