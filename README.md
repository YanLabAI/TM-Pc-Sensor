# Classification and Regression Models

A Python implementation of machine learning models for dual prediction tasks: classification and regression using various ensemble methods and support vector machines.

## 📋 Overview

This project implements machine learning pipelines for:
- **Classification Task**: Predicting categorical labels (e.g., sensitivity classes)
- **Regression Task**: Predicting continuous values (e.g., adsorption energy, temperature, etc.)

Both tasks use the same feature set but different target variables, making it ideal for multi-target prediction scenarios.

## 🚀 Features

- **Multiple ML Models**:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - Support Vector Machines (SVM/SVR)

- **Robust Evaluation**:
  - 5-fold cross-validation on training data
  - Independent test set evaluation
  - Comprehensive metrics for both tasks

- **Automated Pipeline**:
  - Data loading and preprocessing
  - Stratified splitting for classification
  - Model training and evaluation
  - Performance reporting

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn xgboost openpyxl
```

### Dependency Versions
- Python >= 3.7
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- openpyxl >= 3.0.0 (for Excel file support)

## 📊 Data Format

Your data file should be structured as follows:

| ID | Info1 | Info2 | ... | sensitivity_class | target_value | Feature1 | Feature2 | Feature3 | ... |
|----|-------|-------|-----|-------------------|--------------|----------|----------|----------|-----|
| 1  | ...   | ...   | ... | 0                 | -1.23        | 0.45     | 3.21     | 1.67     | ... |
| 2  | ...   | ...   | ... | 1                 | -0.87        | 0.76     | 2.89     | 2.34     | ... |
| 3  | ...   | ...   | ... | 0                 | -1.45        | 0.34     | 4.12     | 1.89     | ... |

**Key Points**:
- Features should be in consecutive columns starting from a specific index (default: column 5)
- Classification target: categorical values (0, 1, 2, ...)
- Regression target: continuous numerical values
- Both targets should be in separate columns

## 🛠️ Usage

### 1. Prepare Your Data

Replace the placeholder names in `CLF_REG.py`:

```python
# Line 40 and 118: Update file path
data_cls = pd.read_excel('data.xlsx')  # Change to your file path

# Line 47: Update classification target column name
y_cls = data_cls['sensitivity_class']  # Change to your column name

# Line 125: Update regression target column name
y_reg = data_reg['target_value']  # Change to your column name

# Line 44 and 122: Adjust feature column index if needed
X_cls = data_cls.iloc[:, 5:]  # Change 5 to your starting column index
```

### 2. Run the Script

```bash
python CLF_REG.py
```

### 3. Interpret Results

The script will output:

**Classification Metrics**:
- Accuracy: Overall correct prediction rate
- F1-score: Harmonic mean of precision and recall
- Recall (Sensitivity): True positive rate
- Precision: Positive predictive value

**Regression Metrics**:
- R²: Coefficient of determination (1.0 = perfect fit)
- RMSE: Root Mean Squared Error (lower is better)
- MAE: Mean Absolute Error (lower is better)


