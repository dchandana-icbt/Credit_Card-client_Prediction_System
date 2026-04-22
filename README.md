# Credit Card Client Fraud Prediction System

## Project Overview

This project is a Python-based data analysis and machine learning solution developed for the **Programming for Data Analysis** module. The aim of the project is to analyse credit card client data and build predictive models to identify whether a client is likely to become fraudulent.

The project follows an end-to-end data analytics workflow, including:

- Data loading and merging
- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Feature engineering
- Model building
- Model improvement
- Model evaluation
- Streamlit GUI application development
- Version control using GitHub

The target variable is `TARGET`, where:

| Target Value | Meaning |
|---:|---|
| `0` | Non-fraudulent client |
| `1` | Fraudulent client |

Although the assignment wording refers to regression, the target variable is binary. Therefore, this project is treated as a **binary classification problem**.

---

## Repository

GitHub Repository:

```text
https://github.com/dchandana-icbt/Credit_Card-client_Prediction_System
```

---

## Dataset Description

The original assignment dataset was provided as two CSV files:

1. `Credir_Card_Dataset_2025_Sept_1.csv`
2. `Credir_Card_Dataset_2025_Sept_2.csv`

The datasets were merged using a common client identifier. One dataset used `ID`, while the other used `User`. The `User` column was renamed to `ID` before merging.

The final merged dataset contains:

- **25,134 records**
- **19 original variables**
- Numerical, categorical, and binary variables
- Highly imbalanced target variable

Main features include:

| Feature | Description |
|---|---|
| `ID` | Unique client identifier |
| `GENDER` | Client gender |
| `CAR` | Car ownership |
| `REALITY` | Property ownership |
| `NO_OF_CHILD` | Number of children |
| `FAMILY_TYPE` | Family status |
| `HOUSE_TYPE` | Housing type |
| `FLAG_MOBIL` | Mobile phone indicator |
| `WORK_PHONE` | Work phone indicator |
| `PHONE` | Personal phone indicator |
| `E_MAIL` | Email availability |
| `FAMILY SIZE` | Total family size |
| `BEGIN_MONTH` | Client record duration indicator |
| `AGE` | Client age |
| `YEARS_EMPLOYED` | Employment duration |
| `TARGET` | Fraud label |
| `INCOME` | Client income |
| `INCOME_TYPE` | Type of income |
| `EDUCATION_TYPE` | Education level |

---

## Project Folder Structure

```text
Credit_Card-client_Prediction_System/
│
├── data/
│   ├── raw/
│   │   ├── Credir_Card_Dataset_2025_Sept_1.csv
│   │   └── Credir_Card_Dataset_2025_Sept_2.csv
│   │
│   ├── interim/
│   │   └── Merged_Credit_Card_Dataset.csv
│   │
│   └── processed/
│       ├── Final_Cleaned_Credit_Card_Dataset.csv
│       ├── Model_Ready_Base_Credit_Card_Dataset.csv
│       ├── Model_Ready_Encoded_Credit_Card_Dataset.csv
│       └── splits/
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           ├── y_test.csv
│           ├── X_train_normalized.csv
│           ├── X_test_normalized.csv
│           └── minmax_scaler.pkl
│
├── reports/
│   ├── data_preprocessing/
│   │   ├── basic_column_summary.csv
│   │   ├── categorical_summary.csv
│   │   ├── numerical_summary.csv
│   │   ├── data_types.csv
│   │   ├── dataset_head.csv
│   │   ├── missing_values_before_cleaning.csv
│   │   ├── missing_values_after_cleaning.csv
│   │   ├── unique_values_before_cleaning.csv
│   │   ├── correlation_matrix.csv
│   │   ├── correlation_with_target.csv
│   │   ├── eda_report.txt
│   │   └── charts/
│   │
│   └── model_results/
│       ├── logistic_regression_model.pkl
│       ├── ann_model.pkl
│       ├── logistic_regression_classification_report.txt
│       ├── ann_classification_report.txt
│       ├── logistic_regression_confusion_matrix.png
│       ├── ann_confusion_matrix.png
│       ├── logistic_regression_coefficients.csv
│       ├── model_comparison_results.csv
│       ├── model_comparison_chart.png
│       ├── best_model_parameters.csv
│       ├── logistic_regression_thresholds.png
│       └── ann_thresholds.png
│
├── src/
│   └── ml_project/
│       ├── data_loader.py
│       ├── preprocess.py
│       ├── train_models.py
│       ├── train_models_improved.py
│       └── streamlit_credit_fraud_app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Main Python Scripts

### 1. `data_loader.py`

This script is used to load the two original datasets and merge them using the common client identifier.

Main responsibilities:

- Load raw CSV files
- Inspect dataset structure
- Rename `User` to `ID`
- Merge datasets
- Save merged dataset into `data/interim`

Output:

```text
data/interim/Merged_Credit_Card_Dataset.csv
```

---

### 2. `preprocess.py`

This script performs EDA, preprocessing, feature engineering, train-test splitting, and normalization.

Main responsibilities:

- Load merged dataset
- Identify rows and columns
- Identify data types
- Check missing values
- Check unique values
- Check duplicate records
- Analyse target distribution
- Handle missing values
- Remove duplicate records
- Remove constant columns
- Cap outliers using IQR
- Create engineered features
- Encode categorical variables
- Split data into train and test sets
- Normalize features using MinMaxScaler
- Save processed datasets and reports

Important engineered features:

| Feature | Description |
|---|---|
| `TOTAL_CONTACTS` | Sum of phone/email contact indicators |
| `HAS_CAR` | Binary version of car ownership |
| `HAS_PROPERTY` | Binary version of property ownership |
| `INCOME_BAND` | Income grouped into bands |
| `INCOME_PER_PERSON` | Income divided by family size |
| `CHILD_RATIO` | Number of children divided by family size |

Important outputs:

```text
data/processed/Final_Cleaned_Credit_Card_Dataset.csv
data/processed/Model_Ready_Base_Credit_Card_Dataset.csv
data/processed/Model_Ready_Encoded_Credit_Card_Dataset.csv
data/processed/splits/X_train.csv
data/processed/splits/X_test.csv
data/processed/splits/y_train.csv
data/processed/splits/y_test.csv
data/processed/splits/X_train_normalized.csv
data/processed/splits/X_test_normalized.csv
data/processed/splits/minmax_scaler.pkl
```

---

### 3. `train_models.py`

This script trains the initial machine learning models.

Models included:

1. Logistic Regression
2. Artificial Neural Network

Main responsibilities:

- Load normalized training and testing data
- Train Logistic Regression model
- Train ANN model
- Evaluate models
- Save classification reports
- Save confusion matrices
- Save trained `.pkl` models
- Save model comparison results

Important outputs:

```text
reports/model_results/logistic_regression_model.pkl
reports/model_results/ann_model.pkl
reports/model_results/logistic_regression_classification_report.txt
reports/model_results/ann_classification_report.txt
reports/model_results/logistic_regression_confusion_matrix.png
reports/model_results/ann_confusion_matrix.png
reports/model_results/model_comparison_results.csv
reports/model_results/model_comparison_chart.png
```

---

### 4. `train_models_improved.py`

This script improves the model training process.

Improvement techniques used:

- SMOTE for class imbalance handling
- Stratified K-Fold cross-validation
- GridSearchCV hyperparameter tuning
- Threshold optimisation
- F1-score based model selection
- Precision-recall threshold plots

Main responsibilities:

- Load normalized training and testing data
- Apply SMOTE to the training set
- Tune Logistic Regression using GridSearchCV
- Tune ANN using GridSearchCV
- Use Stratified K-Fold cross-validation
- Optimise classification threshold
- Save improved models and reports

Important outputs:

```text
reports/model_results/logistic_regression_model.pkl
reports/model_results/ann_model.pkl
reports/model_results/best_model_parameters.csv
reports/model_results/logistic_regression_thresholds.png
reports/model_results/ann_thresholds.png
reports/model_results/model_comparison_results.csv
```

---

### 5. `streamlit_credit_fraud_app.py`

This script provides a graphical user interface using Streamlit.

Main features:

- Predict fraud probability for a new client
- Select between Logistic Regression and ANN
- Enter client profile through the UI
- Automatically preprocess user input
- Align input features with training columns
- Apply saved scaler
- Load saved `.pkl` models
- Display prediction result as a percentage
- Show EDA report section

Application sections:

| Section | Purpose |
|---|---|
| Prediction | Enter new client details and predict fraud probability |
| EDA Report | Display dataset overview, preprocessing summary, and charts |

---

## Exploratory Data Analysis Summary

The EDA process focused on three areas:

### 1. Fundamental Data Understanding

The project analysed:

- Number of rows and columns
- Data types
- Unique values
- Missing values
- Duplicate records
- Target distribution

Important findings:

- Dataset has 25,134 records.
- Dataset contains numerical, categorical, and binary variables.
- Missing values were found in `FAMILY SIZE`, `YEARS_EMPLOYED`, and `INCOME_TYPE`.
- `FLAG_MOBIL` had only one unique value and was not useful for prediction.
- The target variable is highly imbalanced.

### 2. Data Preprocessing

The preprocessing stage included:

- Missing value handling
- Duplicate removal
- Constant column removal
- Outlier treatment using IQR
- Feature engineering
- Categorical encoding
- Train-test splitting
- Normalization

### 3. Statistical Analysis and Visualisation

The project generated:

- Target distribution chart
- Income distribution chart
- Age distribution chart
- Years employed boxplot
- Income type distribution chart
- Income by target chart
- Years employed by target chart
- Fraud rate by house type chart
- Age vs income by target chart
- Correlation heatmap

---

## Model Building Summary

Two models were selected:

### Logistic Regression

Logistic Regression was used as a baseline classification model.

Strengths:

- Simple
- Interpretable
- Suitable for binary classification
- Provides feature coefficients

Limitations:

- May not capture complex nonlinear patterns
- Sensitive to class imbalance
- Weaker fraud-class recall

### Artificial Neural Network

ANN was used as a more flexible model that can learn nonlinear relationships.

Strengths:

- Can learn complex feature relationships
- Performed better than Logistic Regression
- Achieved higher precision, recall, F1-score, and ROC-AUC

Limitations:

- Less interpretable
- Requires tuning
- Still affected by class imbalance

---

## Final Model Comparison

| Metric | Improved Logistic Regression | Improved ANN |
|---|---:|---:|
| Accuracy | 0.9624 | 0.9783 |
| Precision | 0.1053 | 0.3134 |
| Recall | 0.1667 | 0.2500 |
| F1-score | 0.1290 | 0.2781 |
| ROC-AUC | 0.6954 | 0.7346 |

The ANN model performed better than Logistic Regression across all major evaluation metrics.

However, both models still showed low fraud-class recall because the dataset is highly imbalanced.

---

## Key Evaluation Findings

Accuracy alone is not sufficient for this project because the dataset is highly imbalanced. A model can achieve high accuracy by predicting most clients as non-fraudulent.

Therefore, the project used:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The ANN model is the better-performing model overall, but further improvement is still required to increase fraud detection performance.

---

## Streamlit Application

The Streamlit application allows users to interact with the trained models.

To run the app:

```bash
python3 -m streamlit run src/ml_project/streamlit_credit_fraud_app.py
```

The app allows the user to enter:

- Gender
- Age
- Income
- Years employed
- Begin month
- Number of children
- Family size
- Family type
- House type
- Income type
- Education type
- Car ownership
- Property ownership
- Phone availability

The app then predicts the fraud probability using the selected model.

---

## Installation

Create and activate a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If Streamlit or imbalanced-learn is missing, install them manually:

```bash
pip install streamlit imbalanced-learn
```

---

## How to Run the Project

### Step 1: Load and Merge Data

```bash
python src/ml_project/data_loader.py
```

### Step 2: Run Preprocessing and EDA

```bash
python src/ml_project/preprocess.py
```

### Step 3: Train Initial Models

```bash
python src/ml_project/train_models.py
```

### Step 4: Train Improved Models

```bash
python src/ml_project/train_models_improved.py
```

### Step 5: Run Streamlit Application

```bash
python3 -m streamlit run src/ml_project/streamlit_credit_fraud_app.py
```

---

## Output Files

### EDA Outputs

```text
reports/data_preprocessing/
```

Contains:

- Summary CSV files
- Missing value reports
- Correlation reports
- EDA charts
- EDA text report

### Model Outputs

```text
reports/model_results/
```

Contains:

- Classification reports
- Confusion matrices
- Model comparison results
- Threshold plots
- Saved `.pkl` models

### Processed Data

```text
data/processed/
```

Contains:

- Cleaned dataset
- Model-ready dataset
- Encoded dataset
- Train-test split files
- Normalized files
- Saved scaler

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical processing |
| Matplotlib | Visualisation |
| Seaborn | Statistical charts |
| scikit-learn | Machine learning |
| imbalanced-learn | SMOTE class balancing |
| Streamlit | GUI application |
| joblib | Saving models and scaler |
| GitHub | Version control |

---

## Version Control

This project is version-controlled using Git and GitHub. The repository contains commit history showing project development progress.

Version control supports:

- Tracking code changes
- Managing project files
- Documenting progress
- Satisfying Task 5 of the assignment

---

## Limitations

The project has some limitations:

- The dataset is highly imbalanced.
- Fraud-class recall remains low.
- Logistic Regression may not capture complex relationships.
- ANN is less interpretable.
- The model predicts client-level risk, not real-time transaction fraud.
- Further external validation would be required before real-world deployment.

---

## Future Improvements

Recommended future improvements:

- Add XGBoost, LightGBM, or CatBoost
- Use advanced resampling methods such as SMOTEENN or ADASYN
- Add SHAP explainability
- Improve feature engineering
- Add cost-sensitive threshold selection
- Add Precision-Recall AUC
- Add model performance tab to the Streamlit app
- Add downloadable prediction reports
- Deploy the app using Streamlit Cloud

---

## Conclusion

This project successfully developed an end-to-end credit card client fraud prediction system. It includes data loading, dataset merging, exploratory data analysis, preprocessing, feature engineering, model building, model improvement, model evaluation, and Streamlit application development.

The ANN model performed better than Logistic Regression across all major metrics, but both models were affected by the highly imbalanced dataset. Therefore, the project highlights the importance of using precision, recall, F1-score, ROC-AUC, and confusion matrices instead of relying only on accuracy.

Overall, the project demonstrates a complete programming-driven data analysis and machine learning workflow for a real-world financial fraud detection problem.
