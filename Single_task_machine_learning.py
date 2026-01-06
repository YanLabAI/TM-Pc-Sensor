"""
Classification and Regression Models for Sensitivity and Adsorption Energy Prediction

This script implements machine learning models for two tasks:
1. Classification: Predicting sensitivity (categorical)
2. Regression: Predicting adsorption energy (continuous)

Models used: Random Forest, XGBoost, Gradient Boosting, SVM
Evaluation: 5-fold cross-validation on training set + final test set evaluation

==============================================================================
USAGE INSTRUCTIONS:
==============================================================================
Before running this script, you need to customize the following:

1. DATA FILE:
   - Replace 'data.xlsx' with your actual data file path (lines 40, 118)
   - Supported formats: Excel (.xlsx, .xls) or CSV (.csv)
   - For CSV files, change pd.read_excel() to pd.read_csv()

2. FEATURE COLUMNS:
   - Update the column index in iloc[:, 5:] if your features don't start at column 5
   - Example: If features start at column 3, use iloc[:, 3:]

3. TARGET COLUMNS:
   - Classification: Replace 'sensitivity_class' with your classification target column name (line 47)
   - Regression: Replace 'target_value' with your regression target column name (line 125)

4. EXAMPLE DATA STRUCTURE:
   Your Excel/CSV file should look like:
   | ID | Info1 | Info2 | ... | sensitivity_class | target_value | Feature1 | Feature2 | ... |
   |  1 |  ...  |  ...  | ... |         0         |     -1.23    |   0.45   |   3.21   | ... |
   |  2 |  ...  |  ...  | ... |         1         |     -0.87    |   0.76   |   2.89   | ... |
   
   In this example:
   - Column index 5 onwards are features (Feature1, Feature2, ...)
   - 'sensitivity_class' is the classification target
   - 'target_value' is the regression target

==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

# --- Import models ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# --- Import utilities and metrics ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             r2_score, mean_squared_error, mean_absolute_error)


def main():
    # ==============================================================================
    # Part 1: Classification Model (Sensitivity)
    # ==============================================================================
    print("=" * 60)
    print(" Part 1: Classification Model (Predicting Sensitivity)")
    print("=" * 60)

    try:
        # 1. Load and prepare data
        # TODO: Replace 'data.xlsx' with your actual data file path
        data_cls = pd.read_excel('data.xlsx')  # Replace with your file: 'your_data.xlsx' or 'your_data.csv'
        
        # TODO: Adjust the column index based on your data structure
        # Features start from column index 5 (adjust if your features start at a different column)
        X_cls = data_cls.iloc[:, 5:]  # Select all columns from index 5 onwards as features
        
        # TODO: Replace 'sensitivity_class' with your actual classification target column name
        y_cls = data_cls['sensitivity_class']  # Target: classification label (e.g., 0, 1, 2...)
        
        print(f"Classification task: Total {X_cls.shape[0]} samples loaded.\n")

        # 2. Split data into training and test sets (80% train, 20% test)
        # Use stratify=y_cls to ensure class proportions are maintained
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
        print(f"Data split: {X_train_cls.shape[0]} training samples and "
              f"{X_test_cls.shape[0]} test samples.\n")
        
        # 3. Define classification models
        classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, 
                                         eval_metric='logloss'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': make_pipeline(StandardScaler(), SVC(random_state=42))
        }
        
        # 4. Perform 5-fold cross-validation on training set
        print("--- Step 1: 5-Fold Cross-Validation on Training Set ---")
        cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_cls = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']

        for name, model in classifiers.items():
            # cross_validate can compute multiple metrics simultaneously
            cv_results = cross_validate(model, X_train_cls, y_train_cls, 
                                       cv=cv_stratified, scoring=scoring_cls)
            print(f"\nModel: {name} (Cross-Validation Average Results)")
            print(f"  - Mean Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}")
            print(f"  - Mean F1-score:  {np.mean(cv_results['test_f1_macro']):.4f}")
            print(f"  - Mean Recall:    {np.mean(cv_results['test_recall_macro']):.4f}")
            print(f"  - Mean Precision: {np.mean(cv_results['test_precision_macro']):.4f}")
            
        # 5. Train on full training set and evaluate on test set
        print("\n\n--- Step 2: Final Performance Evaluation on Test Set ---")
        for name, model in classifiers.items():
            # Train model
            model.fit(X_train_cls, y_train_cls)
            # Predict on test set
            y_pred_cls = model.predict(X_test_cls)
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test_cls, y_pred_cls)
            f1 = f1_score(y_test_cls, y_pred_cls, average='macro', zero_division=0)
            recall = recall_score(y_test_cls, y_pred_cls, average='macro', zero_division=0)
            precision = precision_score(y_test_cls, y_pred_cls, average='macro', 
                                       zero_division=0)
            
            print(f"\nModel: {name} (Final Performance on Test Set)")
            print(f"  - Accuracy:  {accuracy:.4f}")
            print(f"  - F1-score:  {f1:.4f}")
            print(f"  - Recall:    {recall:.4f}")
            print(f"  - Precision: {precision:.4f}")

    except FileNotFoundError:
        print("Error: 'data.xlsx' file not found. Please ensure the file is in the same "
              "directory as the script, or provide the full file path.")
    except KeyError as e:
        print(f"Error: Column {e} not found in data. Please verify the target column "
              f"name matches your classification target column (e.g., 'sensitivity_class').")


    # ==============================================================================
    # Part 2: Regression Model (Adsorption Energy)
    # ==============================================================================
    print("\n\n" + "=" * 60)
    print(" Part 2: Regression Model (Predicting Adsorption Energy)")
    print("=" * 60)

    try:
        # 1. Load and prepare data
        # TODO: Replace 'data.xlsx' with your actual data file path
        data_reg = pd.read_excel('data.xlsx')  # Replace with your file: 'your_data.xlsx' or 'your_data.csv'
        
        # TODO: Adjust the column index based on your data structure
        # Features start from column index 5 (adjust if your features start at a different column)
        X_reg = data_reg.iloc[:, 5:]  # Select all columns from index 5 onwards as features
        
        # TODO: Replace 'target_value' with your actual regression target column name
        y_reg = data_reg['target_value']  # Target: continuous value (e.g., energy, price, score...)
        
        print(f"Regression task: Total {X_reg.shape[0]} samples loaded.\n")
        
        # 2. Split data into training and test sets
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        print(f"Data split: {X_train_reg.shape[0]} training samples and "
              f"{X_test_reg.shape[0]} test samples.\n")

        # 3. Define regression models
        regressors = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': make_pipeline(StandardScaler(), SVR())
        }

        # 4. Perform 5-fold cross-validation on training set
        print("--- Step 1: 5-Fold Cross-Validation on Training Set ---")
        cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring_reg = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
        
        for name, model in regressors.items():
            cv_results = cross_validate(model, X_train_reg, y_train_reg, 
                                       cv=cv_kfold, scoring=scoring_reg)
            print(f"\nModel: {name} (Cross-Validation Average Results)")
            print(f"  - Mean R²:   {np.mean(cv_results['test_r2']):.4f}")
            print(f"  - Mean RMSE: {-np.mean(cv_results['test_neg_root_mean_squared_error']):.4f}")
            print(f"  - Mean MAE:  {-np.mean(cv_results['test_neg_mean_absolute_error']):.4f}")
            
        # 5. Train on full training set and evaluate on test set
        print("\n\n--- Step 2: Final Performance Evaluation on Test Set ---")
        for name, model in regressors.items():
            # Train model
            model.fit(X_train_reg, y_train_reg)
            # Predict on test set
            y_pred_reg = model.predict(X_test_reg)
            
            # Calculate evaluation metrics
            r2 = r2_score(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            
            print(f"\nModel: {name} (Final Performance on Test Set)")
            print(f"  - R²:   {r2:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - MAE:  {mae:.4f}")

    except FileNotFoundError:
        print("Error: 'data.xlsx' file not found. Please ensure the file is in the same "
              "directory as the script, or provide the full file path.")
    except KeyError as e:
        print(f"Error: Column {e} not found in data. Please verify the target column "
              f"name matches your regression target column (e.g., 'target_value').")


if __name__ == "__main__":
    main()
