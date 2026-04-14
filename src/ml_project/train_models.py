from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier

# =========================================================
# SETTINGS
# =========================================================
project_root = Path(__file__).resolve().parent.parent.parent

SPLITS_DIR = project_root / "data" / "processed" / "splits"
OUTPUT_DIR = project_root / "reports" / "model_results_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logistic Regression and ANN use normalized data
X_train_norm_file = SPLITS_DIR / "X_train_normalized.csv"
X_test_norm_file = SPLITS_DIR / "X_test_normalized.csv"

# XGBoost uses non-normalized data
X_train_file = SPLITS_DIR / "X_train.csv"
X_test_file = SPLITS_DIR / "X_test.csv"

# Common target files
y_train_file = SPLITS_DIR / "y_train.csv"
y_test_file = SPLITS_DIR / "y_test.csv"

sns.set(style="whitegrid")

# =========================================================
# LOAD DATA
# =========================================================
X_train_norm = pd.read_csv(X_train_norm_file)
X_test_norm = pd.read_csv(X_test_norm_file)

X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)

y_train = pd.read_csv(y_train_file).squeeze("columns").astype(int)
y_test = pd.read_csv(y_test_file).squeeze("columns").astype(int)

print("Data loaded successfully.")
print("X_train_norm shape:", X_train_norm.shape)
print("X_test_norm shape :", X_test_norm.shape)
print("X_train shape     :", X_train.shape)
print("X_test shape      :", X_test.shape)
print("y_train shape     :", y_train.shape)
print("y_test shape      :", y_test.shape)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        roc = roc_auc_score(y_true, y_proba)
    else:
        roc = roc_auc_score(y_true, y_pred)

    return pd.DataFrame(
        [[model_name, acc, prec, rec, f1, roc]],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    )


def save_classification_reports(model_name, y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)

    report_df = pd.DataFrame(report_dict).transpose()
    safe_name = model_name.lower().replace(" ", "_")

    report_df.to_csv(OUTPUT_DIR / f"{safe_name}_classification_report.csv", index=True)

    with open(OUTPUT_DIR / f"{safe_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_text)

    return report_df, report_text


def plot_conf_matrix(model_name, y_true, y_pred, file_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name)
    plt.close()


def build_report_paragraph(results_df):
    best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
    best_precision = results_df.loc[results_df["Precision"].idxmax()]
    best_recall = results_df.loc[results_df["Recall"].idxmax()]
    best_f1 = results_df.loc[results_df["F1 Score"].idxmax()]
    best_roc = results_df.loc[results_df["ROC-AUC"].idxmax()]

    paragraph = (
        f"The three classification models, namely Logistic Regression, ANN, and XGBoost, "
        f"were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. "
        f"Among the tested models, {best_accuracy['Model']} achieved the highest accuracy "
        f"({best_accuracy['Accuracy']:.4f}), while {best_precision['Model']} produced the best precision "
        f"({best_precision['Precision']:.4f}). In terms of fraud detection sensitivity, "
        f"{best_recall['Model']} achieved the highest recall ({best_recall['Recall']:.4f}), which is particularly important "
        f"for an imbalanced fraud detection problem. Similarly, {best_f1['Model']} obtained the highest F1-score "
        f"({best_f1['F1 Score']:.4f}), indicating the best balance between precision and recall. "
        f"Based on ROC-AUC, {best_roc['Model']} showed the strongest overall class discrimination ability "
        f"with a score of {best_roc['ROC-AUC']:.4f}. Overall, the results suggest that while all three models are capable "
        f"of learning patterns from the dataset, the most suitable model should be selected by giving more importance to recall, "
        f"F1-score, and ROC-AUC rather than accuracy alone, due to the highly imbalanced nature of the fraud detection dataset."
    )
    return paragraph


# =========================================================
# 1. LOGISTIC REGRESSION
# =========================================================
log_model = LogisticRegression(
    random_state=2,
    max_iter=2000,
    class_weight="balanced"
)
log_model.fit(X_train_norm, y_train)

y_pred_log = log_model.predict(X_test_norm)
y_proba_log = log_model.predict_proba(X_test_norm)[:, 1]

results_log = evaluate_model("Logistic Regression", y_test, y_pred_log, y_proba_log)
log_report_df, log_report_text = save_classification_reports("Logistic Regression", y_test, y_pred_log)

plot_conf_matrix(
    "Logistic Regression",
    y_test,
    y_pred_log,
    "logistic_regression_confusion_matrix.png"
)

log_coefficients = pd.DataFrame({
    "Feature": X_train_norm.columns,
    "Coefficient": log_model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)
log_coefficients.to_csv(OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False)

print("\nLogistic Regression Results")
print(results_log)
print("\nLogistic Regression Classification Report")
print(log_report_text)


# =========================================================
# 2. ANN CLASSIFIER
# =========================================================
ann_model = MLPClassifier(
    hidden_layer_sizes=(200,),
    max_iter=1000,
    random_state=42
)
ann_model.fit(X_train_norm, y_train)

y_pred_ann = ann_model.predict(X_test_norm)
y_proba_ann = ann_model.predict_proba(X_test_norm)[:, 1]

results_ann = evaluate_model("ANN", y_test, y_pred_ann, y_proba_ann)
ann_report_df, ann_report_text = save_classification_reports("ANN", y_test, y_pred_ann)

plot_conf_matrix(
    "ANN",
    y_test,
    y_pred_ann,
    "ann_confusion_matrix.png"
)

print("\nANN Results")
print(results_ann)
print("\nANN Classification Report")
print(ann_report_text)


# =========================================================
# 3. XGBOOST
# =========================================================
# negative_count = (y_train == 0).sum()
# positive_count = (y_train == 1).sum()
# scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

# xgb_model = XGBClassifier(
#     n_estimators=300,
#     max_depth=5,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight=scale_pos_weight,
#     eval_metric="logloss",
#     random_state=42
# )
# xgb_model.fit(X_train, y_train)

# y_pred_xgb = xgb_model.predict(X_test)
# y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# results_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
# xgb_report_df, xgb_report_text = save_classification_reports("XGBoost", y_test, y_pred_xgb)

# plot_conf_matrix(
#     "XGBoost",
#     y_test,
#     y_pred_xgb,
#     "xgboost_confusion_matrix.png"
# )

# xgb_importance = pd.DataFrame({
#     "Feature": X_train.columns,
#     "Importance": xgb_model.feature_importances_
# }).sort_values(by="Importance", ascending=False)
# xgb_importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)

# print("\nXGBoost Results")
# print(results_xgb)
# print("\nXGBoost Classification Report")
# print(xgb_report_text)


# =========================================================
# FINAL COMPARISON
# =========================================================
# final_results = pd.concat([results_log, results_ann, results_xgb], ignore_index=True)
final_results = pd.concat([results_log, results_ann], ignore_index=True)
final_results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

print("\nFinal Model Comparison")
print(final_results)

# Comparison chart
plt.figure(figsize=(10, 6))
comparison_plot = final_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
comparison_plot.plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_chart.png")
plt.close()

# Report-ready paragraph
comparison_paragraph = build_report_paragraph(final_results)

with open(OUTPUT_DIR / "report_ready_model_comparison_paragraph.txt", "w", encoding="utf-8") as f:
    f.write(comparison_paragraph)

print("\nReport-ready comparison paragraph:\n")
print(comparison_paragraph)

print(f"\nAll results saved successfully in: {OUTPUT_DIR}")