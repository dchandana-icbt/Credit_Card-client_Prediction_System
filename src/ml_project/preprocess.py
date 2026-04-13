import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Get the project root directory (assuming this script is in src/ml_project/)
project_root = Path(__file__).resolve().parent.parent.parent

# =========================================================
# SETTINGS
# =========================================================
INPUT_FILE = project_root / "data" / "interim" / "Merged_Credit_Card_Dataset.csv"
OUTPUT_DIR = project_root / "reports" / "data_preprocessing"
PROCESSED_DIR = project_root / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORT_FILE = OUTPUT_DIR / "eda_report.txt"
FINAL_CLEAN_FILE = PROCESSED_DIR / "Final_Cleaned_Credit_Card_Dataset.csv"
MODEL_BASE_FILE = PROCESSED_DIR / "Model_Ready_Base_Credit_Card_Dataset.csv"
MODEL_ENCODED_FILE = PROCESSED_DIR / "Model_Ready_Encoded_Credit_Card_Dataset.csv"

# Train/test split outputs
X_TRAIN_FILE = SPLIT_DIR / "X_train.csv"
X_TEST_FILE = SPLIT_DIR / "X_test.csv"
Y_TRAIN_FILE = SPLIT_DIR / "y_train.csv"
Y_TEST_FILE = SPLIT_DIR / "y_test.csv"

# Normalized train/test split outputs
X_TRAIN_NORMALIZED_FILE = SPLIT_DIR / "X_train_normalized.csv"
X_TEST_NORMALIZED_FILE = SPLIT_DIR / "X_test_normalized.csv"
SCALER_FILE = SPLIT_DIR / "minmax_scaler.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def cap_outliers_iqr(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap outliers using the IQR rule."""
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)
    return dataframe


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE)
raw_df = df.copy()

# =========================================================
# ABOUT DATA
# =========================================================
n_rows, n_cols = df.shape
duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["ID"].duplicated().sum()) if "ID" in df.columns else 0
missing_values = df.isnull().sum()
unique_values = df.nunique()

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Save data overview artifacts similar to notebook-style inspection
raw_df.head(10).to_csv(OUTPUT_DIR / "dataset_head.csv", index=False)
missing_values.rename("missing_count").to_csv(OUTPUT_DIR / "missing_values_before_cleaning.csv")
unique_values.rename("unique_values").to_csv(OUTPUT_DIR / "unique_values_before_cleaning.csv")
pd.DataFrame(
    {"column": df.columns, "dtype": [str(df[col].dtype) for col in df.columns]}
).to_csv(OUTPUT_DIR / "data_types.csv", index=False)

# =========================================================
# TARGET DISTRIBUTION
# =========================================================
if "TARGET" not in df.columns:
    raise ValueError("TARGET column not found in the input dataset.")

target_counts = df["TARGET"].value_counts().sort_index()
target_percent = df["TARGET"].value_counts(normalize=True).sort_index() * 100
class_imbalance_ratio = None
if 0 in target_counts.index and 1 in target_counts.index and target_counts.loc[1] != 0:
    class_imbalance_ratio = target_counts.loc[0] / target_counts.loc[1]

# =========================================================
# SAVE BASIC SUMMARY TABLES
# =========================================================
basic_info_df = pd.DataFrame(
    {
        "Column": df.columns,
        "Data_Type": [str(df[col].dtype) for col in df.columns],
        "Missing_Values": [int(df[col].isnull().sum()) for col in df.columns],
        "Unique_Values": [int(df[col].nunique()) for col in df.columns],
    }
)
basic_info_df.to_csv(OUTPUT_DIR / "basic_column_summary.csv", index=False)

df.describe(include=[np.number]).T.to_csv(OUTPUT_DIR / "numerical_summary.csv")
if categorical_cols:
    df.describe(include=["object", "category"]).T.to_csv(OUTPUT_DIR / "categorical_summary.csv")

# =========================================================
# DATA PRE-PROCESSING FOR EDA / CLEAN DATASET
# =========================================================

# 1. Drop constant columns
constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
if constant_columns:
    df.drop(columns=constant_columns, inplace=True)

# 2. Handle missing values
for col in df.select_dtypes(include=["object", "category"]).columns:
    if df[col].isnull().sum() > 0:
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series.iloc[0])

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 3. Remove duplicate rows
before_dedup = len(df)
df.drop_duplicates(inplace=True)
removed_duplicate_rows = before_dedup - len(df)

# 4. Feature engineering
if all(col in df.columns for col in ["WORK_PHONE", "PHONE"]):
    email_component = df["E_MAIL"] if "E_MAIL" in df.columns else 0
    df["TOTAL_CONTACTS"] = df["WORK_PHONE"] + df["PHONE"] + email_component

if "CAR" in df.columns:
    df["HAS_CAR"] = df["CAR"].map({"Y": 1, "N": 0})

if "REALITY" in df.columns:
    df["HAS_PROPERTY"] = df["REALITY"].map({"Y": 1, "N": 0})

if "INCOME" in df.columns:
    df["INCOME_BAND"] = pd.cut(
        df["INCOME"],
        bins=[-np.inf, 100000, 200000, 300000, 500000, np.inf],
        labels=["Low", "Lower_Middle", "Middle", "Upper_Middle", "High"],
    )

if all(col in df.columns for col in ["INCOME", "FAMILY SIZE"]):
    family_size_nonzero = df["FAMILY SIZE"].replace(0, np.nan)
    df["INCOME_PER_PERSON"] = df["INCOME"] / family_size_nonzero
    df["INCOME_PER_PERSON"] = df["INCOME_PER_PERSON"].fillna(df["INCOME_PER_PERSON"].median())

if all(col in df.columns for col in ["NO_OF_CHILD", "FAMILY SIZE"]):
    family_size_nonzero = df["FAMILY SIZE"].replace(0, np.nan)
    df["CHILD_RATIO"] = df["NO_OF_CHILD"] / family_size_nonzero
    df["CHILD_RATIO"] = df["CHILD_RATIO"].fillna(0)

# 5. Outlier capping using IQR
outlier_columns = []
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ["ID", "TARGET"]:
        outlier_columns.append(col)
        cap_outliers_iqr(df, col)

# Save post-cleaning missing values too
df.isnull().sum().rename("missing_count_after_cleaning").to_csv(
    OUTPUT_DIR / "missing_values_after_cleaning.csv"
)

# Clean dataset for analysis/reporting
final_clean_df = df.copy()
final_clean_df.to_csv(FINAL_CLEAN_FILE, index=False)

# =========================================================
# MODEL-READY DATASET CHANGES FOR LOGISTIC REGRESSION / XGBOOST
# =========================================================
model_df = final_clean_df.copy()

# Remove non-predictive and redundant columns for modelling
model_drop_columns = [
    "ID",
    "E_MAIL",
    "CAR",
    "REALITY",
    "INCOME_BAND",
]
existing_model_drop_columns = [col for col in model_drop_columns if col in model_df.columns]
if existing_model_drop_columns:
    model_df.drop(columns=existing_model_drop_columns, inplace=True)

# Save base model-ready file with categorical columns retained
model_df.to_csv(MODEL_BASE_FILE, index=False)

# Create encoded version for modelling
encoded_df = model_df.copy()
encoded_categorical_cols = encoded_df.select_dtypes(include=["object", "category"]).columns.tolist()
encoded_df = pd.get_dummies(encoded_df, columns=encoded_categorical_cols, drop_first=True)
encoded_df.to_csv(MODEL_ENCODED_FILE, index=False)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X = encoded_df.drop(columns=["TARGET"])
y = encoded_df["TARGET"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=1,
)

# Save raw train/test splits
X_train.to_csv(X_TRAIN_FILE, index=False)
X_test.to_csv(X_TEST_FILE, index=False)
y_train.to_csv(Y_TRAIN_FILE, index=False)
y_test.to_csv(Y_TEST_FILE, index=False)

train_target_distribution = y_train.value_counts(normalize=True).sort_index() * 100
test_target_distribution = y_test.value_counts(normalize=True).sort_index() * 100

# =========================================================
# DATA NORMALIZATION
# =========================================================
# Apply Min-Max normalization on numeric feature columns using only the training set fit.
# TARGET is not part of X, so only features are normalized.
scaler = MinMaxScaler()

numeric_feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# All encoded features are numeric. This normalization is safe for direct modelling pipelines.
X_train_normalized = X_train.copy()
X_test_normalized = X_test.copy()

X_train_normalized[numeric_feature_cols] = scaler.fit_transform(X_train[numeric_feature_cols])
X_test_normalized[numeric_feature_cols] = scaler.transform(X_test[numeric_feature_cols])

# Save normalized train/test sets and scaler
X_train_normalized.to_csv(X_TRAIN_NORMALIZED_FILE, index=False)
X_test_normalized.to_csv(X_TEST_NORMALIZED_FILE, index=False)
joblib.dump(scaler, SCALER_FILE)

# =========================================================
# ANALYSIS TABLES
# =========================================================
fraud_rate_tables = {}
for col in ["GENDER", "CAR", "REALITY", "FAMILY_TYPE", "HOUSE_TYPE", "INCOME_TYPE", "EDUCATION_TYPE", "INCOME_BAND"]:
    if col in final_clean_df.columns:
        temp = final_clean_df.groupby(col)["TARGET"].agg(["count", "mean"]).reset_index()
        temp.rename(columns={"mean": "fraud_rate"}, inplace=True)
        fraud_rate_tables[col] = temp
        temp.to_csv(OUTPUT_DIR / f"fraud_rate_by_{col}.csv", index=False)

numeric_for_group = [
    c
    for c in [
        "NO_OF_CHILD",
        "FAMILY SIZE",
        "BEGIN_MONTH",
        "AGE",
        "YEARS_EMPLOYED",
        "INCOME",
        "TOTAL_CONTACTS",
        "INCOME_PER_PERSON",
        "CHILD_RATIO",
    ]
    if c in final_clean_df.columns
]
group_means = final_clean_df.groupby("TARGET")[numeric_for_group].mean()
group_means.to_csv(OUTPUT_DIR / "mean_numerical_values_by_target.csv")

corr_df = final_clean_df.select_dtypes(include=[np.number]).corr()
corr_df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

if "TARGET" in corr_df.columns:
    target_corr = corr_df["TARGET"].sort_values(ascending=False)
    target_corr.to_csv(OUTPUT_DIR / "correlation_with_target.csv", header=["correlation_with_target"])

# =========================================================
# VISUALISATIONS
# =========================================================
plt.figure()
sns.countplot(data=final_clean_df, x="TARGET")
plt.title("Distribution of Target Variable")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "01_target_distribution.png")
plt.close()

if "INCOME" in final_clean_df.columns:
    plt.figure()
    sns.histplot(final_clean_df["INCOME"], bins=30, kde=True)
    plt.title("Distribution of Income")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_income_distribution.png")
    plt.close()

if "AGE" in final_clean_df.columns:
    plt.figure()
    sns.histplot(final_clean_df["AGE"], bins=30, kde=True)
    plt.title("Distribution of Age")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_age_distribution.png")
    plt.close()

if "YEARS_EMPLOYED" in final_clean_df.columns:
    plt.figure()
    sns.boxplot(data=final_clean_df, y="YEARS_EMPLOYED")
    plt.title("Boxplot of Years Employed")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_years_employed_boxplot.png")
    plt.close()

if "INCOME_TYPE" in final_clean_df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=final_clean_df, x="INCOME_TYPE", order=final_clean_df["INCOME_TYPE"].value_counts().index)
    plt.title("Income Type Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_income_type_distribution.png")
    plt.close()

if "INCOME" in final_clean_df.columns:
    plt.figure()
    sns.boxplot(data=final_clean_df, x="TARGET", y="INCOME")
    plt.title("Income by Target")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_income_by_target.png")
    plt.close()

if "YEARS_EMPLOYED" in final_clean_df.columns:
    plt.figure()
    sns.boxplot(data=final_clean_df, x="TARGET", y="YEARS_EMPLOYED")
    plt.title("Years Employed by Target")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_years_employed_by_target.png")
    plt.close()

if "HOUSE_TYPE" in final_clean_df.columns:
    fraud_house = final_clean_df.groupby("HOUSE_TYPE")["TARGET"].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    fraud_house.plot(kind="bar")
    plt.title("Fraud Rate by House Type")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_fraud_rate_by_house_type.png")
    plt.close()

if "AGE" in final_clean_df.columns and "INCOME" in final_clean_df.columns:
    plt.figure()
    sns.scatterplot(data=final_clean_df, x="AGE", y="INCOME", hue="TARGET", alpha=0.6)
    plt.title("Age vs Income by Target")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "09_age_vs_income_by_target.png")
    plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "10_correlation_heatmap.png")
plt.close()

# =========================================================
# TEXT REPORT
# =========================================================
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("TASK 2 - EDA, PREPROCESSING, TRAIN/TEST SPLIT, AND NORMALIZATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. ABOUT DATA\n")
    f.write(f"Rows: {n_rows}\n")
    f.write(f"Columns: {n_cols}\n")
    f.write(f"Numerical Columns: {len(numerical_cols)}\n")
    f.write(f"Categorical Columns: {len(categorical_cols)}\n")
    f.write(f"Duplicate Rows Before Cleaning: {duplicate_rows}\n")
    f.write(f"Duplicate IDs Before Cleaning: {duplicate_ids}\n")
    f.write(f"Duplicate Rows Removed: {removed_duplicate_rows}\n\n")

    f.write("2. TARGET DISTRIBUTION\n")
    for idx, value in target_counts.items():
        f.write(f"TARGET = {idx}: {value} records ({target_percent[idx]:.3f}%)\n")
    if class_imbalance_ratio is not None:
        f.write(f"Class imbalance ratio (majority/minority): {class_imbalance_ratio:.2f}:1\n")
    f.write("\n")

    f.write("3. CHECKING MISSING VALUES (BEFORE CLEANING)\n")
    for col, val in missing_values.items():
        f.write(f"{col}: {val}\n")
    f.write("\n")

    f.write("4. DATA PRE-PROCESSING PERFORMED\n")
    f.write("- Missing categorical values filled with mode.\n")
    f.write("- Missing numerical values filled with median.\n")
    f.write("- Duplicate rows removed.\n")
    f.write("- Outliers capped using the IQR method.\n")
    f.write("- New features created: TOTAL_CONTACTS, HAS_CAR, HAS_PROPERTY, INCOME_BAND, INCOME_PER_PERSON, CHILD_RATIO.\n")
    f.write("- Clean analysis file saved.\n")
    f.write("- Model-ready base file saved with redundant/non-predictive columns removed.\n")
    f.write("- Encoded model-ready file saved for machine learning.\n\n")

    f.write("5. MODEL-READY COLUMN CHANGES\n")
    if existing_model_drop_columns:
        for col in existing_model_drop_columns:
            f.write(f"- Dropped from model dataset: {col}\n")
    else:
        f.write("No model-specific columns were dropped.\n")
    f.write("\n")

    f.write("6. TRAIN TEST SPLIT\n")
    f.write(f"X_train shape: {X_train.shape}\n")
    f.write(f"X_test shape: {X_test.shape}\n")
    f.write(f"y_train shape: {y_train.shape}\n")
    f.write(f"y_test shape: {y_test.shape}\n")
    f.write("Training target distribution (%):\n")
    for idx, value in train_target_distribution.items():
        f.write(f"  TARGET = {idx}: {value:.3f}%\n")
    f.write("Test target distribution (%):\n")
    for idx, value in test_target_distribution.items():
        f.write(f"  TARGET = {idx}: {value:.3f}%\n")
    f.write("\n")

    f.write("7. DATA NORMALIZATION\n")
    f.write("- MinMaxScaler fitted on training data only.\n")
    f.write(f"- Number of normalized feature columns: {len(numeric_feature_cols)}\n")
    f.write(f"- Scaler saved to: {SCALER_FILE}\n\n")

    f.write("8. NUMERICAL MEAN VALUES BY TARGET\n")
    f.write(group_means.to_string())
    f.write("\n\n")

    f.write("9. CORRELATION WITH TARGET\n")
    if "TARGET" in corr_df.columns:
        f.write(target_corr.to_string())
    f.write("\n\n")

    f.write("10. OUTPUT FILES CREATED\n")
    f.write(f"- Clean analysis dataset: {FINAL_CLEAN_FILE}\n")
    f.write(f"- Model-ready base dataset: {MODEL_BASE_FILE}\n")
    f.write(f"- Model-ready encoded dataset: {MODEL_ENCODED_FILE}\n")
    f.write(f"- X_train: {X_TRAIN_FILE}\n")
    f.write(f"- X_test: {X_TEST_FILE}\n")
    f.write(f"- y_train: {Y_TRAIN_FILE}\n")
    f.write(f"- y_test: {Y_TEST_FILE}\n")
    f.write(f"- X_train_normalized: {X_TRAIN_NORMALIZED_FILE}\n")
    f.write(f"- X_test_normalized: {X_TEST_NORMALIZED_FILE}\n")
    f.write(f"- EDA report: {REPORT_FILE}\n")
    f.write(f"- Charts folder: {CHARTS_DIR}\n")

print("Task 2 preprocessing completed successfully.")
print(f"Clean analysis dataset saved to: {FINAL_CLEAN_FILE}")
print(f"Model-ready base dataset saved to: {MODEL_BASE_FILE}")
print(f"Model-ready encoded dataset saved to: {MODEL_ENCODED_FILE}")
print(f"X_train saved to: {X_TRAIN_FILE}")
print(f"X_test saved to: {X_TEST_FILE}")
print(f"y_train saved to: {Y_TRAIN_FILE}")
print(f"y_test saved to: {Y_TEST_FILE}")
print(f"Normalized X_train saved to: {X_TRAIN_NORMALIZED_FILE}")
print(f"Normalized X_test saved to: {X_TEST_NORMALIZED_FILE}")
print(f"Scaler saved to: {SCALER_FILE}")
print(f"EDA report saved to: {REPORT_FILE}")
print(f"Charts saved in: {CHARTS_DIR}")
