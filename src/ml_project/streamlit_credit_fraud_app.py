from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Risk Predictor", page_icon="💳", layout="wide")
sns.set_style("whitegrid")


# =========================================================
# PATH HELPERS
# =========================================================
def find_project_root(start: Path) -> Path:
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "data" / "processed").exists() and (candidate / "reports").exists():
            return candidate
    return start


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(CURRENT_DIR)

MODEL_DIR = PROJECT_ROOT / "reports" / "model_results"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

LOG_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
ANN_MODEL_PATH = MODEL_DIR / "ann_model.pkl"
SCALER_PATH = SPLITS_DIR / "minmax_scaler.pkl"
X_TRAIN_PATH = SPLITS_DIR / "X_train.csv"
MODEL_BASE_PATH = PROCESSED_DIR / "Model_Ready_Base_Credit_Card_Dataset.csv"
FINAL_CLEAN_PATH = PROCESSED_DIR / "Final_Cleaned_Credit_Card_Dataset.csv"
MERGED_PATH = INTERIM_DIR / "Merged_Credit_Card_Dataset.csv"


# =========================================================
# DATA / MODEL LOADING
# =========================================================
@st.cache_resource
def load_models_and_scaler():
    missing = []
    for path in [LOG_MODEL_PATH, ANN_MODEL_PATH, SCALER_PATH, X_TRAIN_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    logistic_model = joblib.load(LOG_MODEL_PATH)
    ann_model = joblib.load(ANN_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoded_train_columns = pd.read_csv(X_TRAIN_PATH, nrows=1).columns.tolist()
    return logistic_model, ann_model, scaler, encoded_train_columns


@st.cache_data
def load_base_dataset():
    if MODEL_BASE_PATH.exists():
        return pd.read_csv(MODEL_BASE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_eda_dataset():
    if FINAL_CLEAN_PATH.exists():
        return pd.read_csv(FINAL_CLEAN_PATH)
    if MERGED_PATH.exists():
        return pd.read_csv(MERGED_PATH)
    return pd.DataFrame()


# =========================================================
# FEATURE PREPARATION
# =========================================================
def get_category_options(base_df: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if not base_df.empty and column in base_df.columns:
        values = base_df[column].dropna().astype(str).sort_values().unique().tolist()
        if values:
            return values
    return fallback


def build_single_client_dataframe(inputs: dict) -> pd.DataFrame:
    family_size = max(float(inputs["FAMILY SIZE"]), 1.0)
    total_contacts = int(inputs["WORK_PHONE"]) + int(inputs["PHONE"])
    income_per_person = float(inputs["INCOME"]) / family_size
    child_ratio = float(inputs["NO_OF_CHILD"]) / family_size

    row = {
        "GENDER": inputs["GENDER"],
        "NO_OF_CHILD": float(inputs["NO_OF_CHILD"]),
        "FAMILY_TYPE": inputs["FAMILY_TYPE"],
        "HOUSE_TYPE": inputs["HOUSE_TYPE"],
        "WORK_PHONE": float(inputs["WORK_PHONE"]),
        "PHONE": float(inputs["PHONE"]),
        "FAMILY SIZE": family_size,
        "BEGIN_MONTH": float(inputs["BEGIN_MONTH"]),
        "AGE": float(inputs["AGE"]),
        "YEARS_EMPLOYED": float(inputs["YEARS_EMPLOYED"]),
        "INCOME": float(inputs["INCOME"]),
        "INCOME_TYPE": inputs["INCOME_TYPE"],
        "EDUCATION_TYPE": inputs["EDUCATION_TYPE"],
        "TOTAL_CONTACTS": float(total_contacts),
        "HAS_CAR": float(inputs["HAS_CAR"]),
        "HAS_PROPERTY": float(inputs["HAS_PROPERTY"]),
        "INCOME_PER_PERSON": float(income_per_person),
        "CHILD_RATIO": float(child_ratio),
    }
    return pd.DataFrame([row])


def encode_and_align(client_df: pd.DataFrame, encoded_train_columns: list[str]) -> pd.DataFrame:
    encoded_client = pd.get_dummies(client_df)
    encoded_client = encoded_client.reindex(columns=encoded_train_columns, fill_value=0)
    return encoded_client.astype(float)


def scale_only_expected_columns(encoded_client: pd.DataFrame, scaler) -> pd.DataFrame:
    model_input = encoded_client.copy().astype(float)

    if not hasattr(scaler, "feature_names_in_"):
        return model_input

    scaler_columns = [col for col in list(scaler.feature_names_in_) if col in model_input.columns]
    if not scaler_columns:
        return model_input

    scaled_array = scaler.transform(model_input[scaler_columns])
    scaled_part = pd.DataFrame(scaled_array, columns=scaler_columns, index=model_input.index)
    model_input[scaler_columns] = scaled_part.astype(float)

    return model_input.astype(float)


def align_to_model_columns(model_input: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)
        model_input = model_input.reindex(columns=model_columns, fill_value=0)
    return model_input.astype(float)


# =========================================================
# EDA HELPERS
# =========================================================
def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen = {}
    new_cols = []
    for col in out.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    out.columns = new_cols
    return out


def safe_value_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()

    vc = (
        df[column]
        .astype(str)
        .value_counts(dropna=False)
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="Count")
    )
    vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
    return make_unique_columns(vc)


def plot_count(df: pd.DataFrame, column: str, title: str, rotate: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df[column].value_counts().index if column in df.columns else None
    sns.countplot(data=df, x=column, order=order, ax=ax)
    ax.set_title(title)
    if rotate:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_hist(df: pd.DataFrame, column: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_box_by_target(df: pd.DataFrame, column: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="TARGET", y=column, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame, title: str):
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, corr


# =========================================================
# APP LAYOUT
# =========================================================
st.title("💳 Credit Card Fraud Risk Prediction App")
st.caption("Predict fraud probability for a new client and explore the dataset, preprocessing, and EDA results in one interface.")

try:
    logistic_model, ann_model, scaler, encoded_train_columns = load_models_and_scaler()
except Exception as exc:
    st.error("Required model or preprocessing files were not found.")
    st.code(str(exc))
    st.stop()

base_df = load_base_dataset()
eda_df = load_eda_dataset()

fallback_family_types = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
fallback_house_types = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"]
fallback_income_types = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
fallback_education_types = ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]

gender_options = get_category_options(base_df, "GENDER", ["M", "F"])
family_type_options = get_category_options(base_df, "FAMILY_TYPE", fallback_family_types)
house_type_options = get_category_options(base_df, "HOUSE_TYPE", fallback_house_types)
income_type_options = get_category_options(base_df, "INCOME_TYPE", fallback_income_types)
education_type_options = get_category_options(base_df, "EDUCATION_TYPE", fallback_education_types)

prediction_tab, eda_tab = st.tabs(["Prediction", "EDA Report"])


# =========================================================
# PREDICTION TAB
# =========================================================
with prediction_tab:
    with st.sidebar:
        st.header("Client Input")

        gender = st.selectbox("Gender", gender_options)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Annual Income", min_value=0.0, value=150000.0, step=1000.0)
        years_employed = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        begin_month = st.number_input("Begin Month", min_value=0.0, max_value=60.0, value=24.0, step=1.0)
        no_of_child = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)
        family_size = st.number_input("Family Size", min_value=1.0, max_value=20.0, value=2.0, step=1.0)

        family_type = st.selectbox("Family Type", family_type_options)
        house_type = st.selectbox("House Type", house_type_options)
        income_type = st.selectbox("Income Type", income_type_options)
        education_type = st.selectbox("Education Type", education_type_options)

        has_car = st.selectbox("Has Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        has_property = st.selectbox("Has Property", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        work_phone = st.selectbox("Work Phone Available", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        phone = st.selectbox("Personal Phone Available", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        selected_model = st.selectbox("Model Selection", options=["Logistic Regression", "ANN"], index=0)
        predict_button = st.button("Predict Fraud Probability", type="primary")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Entered Client Details")
        preview_df = pd.DataFrame(
            {
                "Field": [
                    "Gender", "Age", "Annual Income", "Years Employed", "Begin Month",
                    "Number of Children", "Family Size", "Family Type", "House Type",
                    "Income Type", "Education Type", "Has Car", "Has Property",
                    "Work Phone", "Personal Phone"
                ],
                "Value": [
                    gender, age, income, years_employed, begin_month,
                    no_of_child, family_size, family_type, house_type,
                    income_type, education_type, "Yes" if has_car else "No",
                    "Yes" if has_property else "No", "Yes" if work_phone else "No",
                    "Yes" if phone else "No"
                ]
            }
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Model Output")
        if predict_button:
            raw_inputs = {
                "GENDER": gender,
                "AGE": age,
                "INCOME": income,
                "YEARS_EMPLOYED": years_employed,
                "BEGIN_MONTH": begin_month,
                "NO_OF_CHILD": no_of_child,
                "FAMILY SIZE": family_size,
                "FAMILY_TYPE": family_type,
                "HOUSE_TYPE": house_type,
                "INCOME_TYPE": income_type,
                "EDUCATION_TYPE": education_type,
                "HAS_CAR": has_car,
                "HAS_PROPERTY": has_property,
                "WORK_PHONE": work_phone,
                "PHONE": phone,
            }

            client_df = build_single_client_dataframe(raw_inputs)
            encoded_client = encode_and_align(client_df, encoded_train_columns)
            model_input = scale_only_expected_columns(encoded_client, scaler)
            model = logistic_model if selected_model == "Logistic Regression" else ann_model
            model_input = align_to_model_columns(model_input, model)

            probability = float(model.predict_proba(model_input)[0, 1])
            prediction = int(probability >= 0.5)

            st.metric(label=f"{selected_model} Fraud Probability", value=f"{probability * 100:.2f}%")
            if prediction == 1:
                st.error("Prediction: Higher fraud risk")
            else:
                st.success("Prediction: Lower fraud risk")

            st.progress(min(max(probability, 0.0), 1.0))
            st.write("### Engineered Features Used")
            st.dataframe(client_df, use_container_width=True, hide_index=True)

            with st.expander("Debug: Prepared Model Input"):
                st.write("Encoded training columns:", len(encoded_train_columns))
                st.write("Scaler columns:", list(getattr(scaler, "feature_names_in_", [])))
                st.write("Model input shape:", model_input.shape)
                st.dataframe(model_input, use_container_width=True, hide_index=True)
        else:
            st.info("Enter a client profile in the sidebar and click Predict Fraud Probability.")


# =========================================================
# EDA TAB
# =========================================================
with eda_tab:
    st.header("Exploratory Data Analysis (EDA) Report")

    if eda_df.empty:
        st.warning("No cleaned or merged dataset was found for EDA display.")
    else:
        n_rows, n_cols = eda_df.shape
        duplicate_rows = int(eda_df.duplicated().sum())
        missing_values = eda_df.isnull().sum().sort_values(ascending=False)
        unique_values = eda_df.nunique().sort_values(ascending=False)
        numerical_cols = eda_df.select_dtypes(include="number").columns.tolist()
        categorical_cols = eda_df.select_dtypes(exclude="number").columns.tolist()

        target_counts = eda_df["TARGET"].value_counts().sort_index() if "TARGET" in eda_df.columns else pd.Series(dtype=int)
        target_percent = (eda_df["TARGET"].value_counts(normalize=True).sort_index() * 100) if "TARGET" in eda_df.columns else pd.Series(dtype=float)

        sub_a, sub_b, sub_c = st.tabs([
            "a) Fundamental Data Understanding",
            "b) Data Preprocessing",
            "c) Statistics and Visualisation",
        ])

        with sub_a:
            st.subheader("Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{n_rows:,}")
            c2.metric("Columns", n_cols)
            c3.metric("Numerical Columns", len(numerical_cols))
            c4.metric("Categorical Columns", len(categorical_cols))

            st.markdown(
                f"""
                **Interpretation:** The dataset contains **{n_rows:,} records** and **{n_cols} variables**, which is adequate for meaningful analysis.
                The data includes both **numerical** and **categorical** features, making it suitable for a full exploratory workflow involving distribution analysis, cleaning, encoding, and model preparation.
                """
            )

            st.subheader("Data Types, Missing Values, and Unique Values")
            summary_df = pd.DataFrame({
                "Column": eda_df.columns,
                "Data Type": [str(eda_df[c].dtype) for c in eda_df.columns],
                "Missing Values": [int(eda_df[c].isnull().sum()) for c in eda_df.columns],
                "Unique Values": [int(eda_df[c].nunique()) for c in eda_df.columns],
            })
            st.dataframe(make_unique_columns(summary_df), use_container_width=True, hide_index=True)

            st.subheader("Missing Value Summary")
            missing_df = missing_values.reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            st.dataframe(make_unique_columns(missing_df), use_container_width=True, hide_index=True)

            if "TARGET" in eda_df.columns and not target_counts.empty:
                st.subheader("Target Distribution")
                td = pd.DataFrame({
                    "Target": target_counts.index,
                    "Count": target_counts.values,
                    "Percentage": [round(v, 3) for v in target_percent.values],
                })
                st.dataframe(make_unique_columns(td), use_container_width=True, hide_index=True)
                st.markdown(
                    f"""
                    **Interpretation:** The target variable is highly imbalanced. Non-fraudulent cases dominate the dataset, while fraudulent cases represent only **{target_percent.get(1, 0):.3f}%** of records.
                    This is important because classification models may otherwise learn to favor the majority class.
                    """
                )

            if categorical_cols:
                st.subheader("Top Categories in Categorical Variables")
                selected_cat = st.selectbox("Choose a categorical column", categorical_cols, key="fund_cat")
                vc = safe_value_counts(eda_df, selected_cat, top_n=10)
                st.dataframe(make_unique_columns(vc), use_container_width=True, hide_index=True)

            st.subheader("Descriptive Statistics")
            if numerical_cols:
                st.dataframe(make_unique_columns(eda_df[numerical_cols].describe().T), use_container_width=True)

        with sub_b:
            st.subheader("Preprocessing Summary")
            st.markdown(
                """
                The main preprocessing workflow carried out on the dataset includes the following steps:

                1. **Handling missing values**
                   - Categorical variables are filled using the **mode**.
                   - Numerical variables are filled using the **median**.

                2. **Removing duplicate entries**
                   - Duplicate rows are checked and removed where necessary.

                3. **Outlier treatment**
                   - Numeric variables are capped using the **IQR rule** to reduce the effect of extreme values.

                4. **Feature engineering**
                   - New variables such as **TOTAL_CONTACTS**, **HAS_CAR**, **HAS_PROPERTY**, **INCOME_PER_PERSON**, and **CHILD_RATIO** are created.

                5. **Cleaning for modelling**
                   - Redundant or non-predictive columns are removed from the model-ready dataset.
                   - Categorical variables are encoded for use in Logistic Regression, ANN, and XGBoost.
                """
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Duplicate Rows", duplicate_rows)
            c2.metric("Columns with Missing Values", int((missing_values > 0).sum()))
            c3.metric("Total Missing Cells", int(missing_values.sum()))

            st.subheader("Data Cleaning Interpretation")
            st.markdown(
                """
                **Interpretation:** The dataset is relatively well-structured, but preprocessing remains necessary to improve model quality and data reliability.
                Missing values must be addressed before modelling, duplicate records should be removed to avoid bias, and outliers should be treated so that extreme values do not distort the analysis.
                Feature engineering further improves the information content of the dataset by deriving more meaningful behavioural and financial indicators.
                """
            )

            if numerical_cols:
                st.subheader("Potential Outlier Review")
                selected_num = st.selectbox("Choose a numerical column for outlier inspection", numerical_cols, key="pre_num")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(data=eda_df, y=selected_num, ax=ax)
                ax.set_title(f"Boxplot of {selected_num}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with sub_c:
            st.subheader("Univariate Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                if "TARGET" in eda_df.columns:
                    st.pyplot(plot_count(eda_df, "TARGET", "Distribution of Target Variable"))
                if "INCOME" in eda_df.columns:
                    st.pyplot(plot_hist(eda_df, "INCOME", "Distribution of Income"))
            with col_b:
                if "AGE" in eda_df.columns:
                    st.pyplot(plot_hist(eda_df, "AGE", "Distribution of Age"))
                if "INCOME_TYPE" in eda_df.columns:
                    st.pyplot(plot_count(eda_df, "INCOME_TYPE", "Income Type Distribution", rotate=True))

            st.markdown(
                """
                **Interpretation:** Univariate analysis shows the distribution of important variables such as target, age, income, and income type.
                The income distribution is usually right-skewed, indicating the presence of higher-income outliers, while categorical variables are dominated by a few major groups.
                """
            )

            st.subheader("Bivariate Analysis")
            col_c, col_d = st.columns(2)
            with col_c:
                if all(col in eda_df.columns for col in ["TARGET", "INCOME"]):
                    st.pyplot(plot_box_by_target(eda_df, "INCOME", "Income by Fraud Status"))
                if all(col in eda_df.columns for col in ["TARGET", "YEARS_EMPLOYED"]):
                    st.pyplot(plot_box_by_target(eda_df, "YEARS_EMPLOYED", "Years Employed by Fraud Status"))
            with col_d:
                if all(col in eda_df.columns for col in ["AGE", "INCOME", "TARGET"]):
                    st.pyplot(plot_scatter(eda_df, "AGE", "INCOME", "TARGET", "Age vs Income by Target"))
                if all(col in eda_df.columns for col in ["HOUSE_TYPE", "TARGET"]):
                    fraud_house = eda_df.groupby("HOUSE_TYPE")["TARGET"].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fraud_house.plot(kind="bar", ax=ax)
                    ax.set_title("Fraud Rate by House Type")
                    ax.set_ylabel("Fraud Rate")
                    ax.tick_params(axis="x", rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            if "TARGET" in eda_df.columns:
                numeric_for_group = [c for c in ["NO_OF_CHILD", "FAMILY SIZE", "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED", "INCOME", "TOTAL_CONTACTS"] if c in eda_df.columns]
                if numeric_for_group:
                    st.subheader("Mean Numerical Values by Target")
                    group_df = eda_df.groupby("TARGET")[numeric_for_group].mean().reset_index()
                    st.dataframe(make_unique_columns(group_df), use_container_width=True)

            st.markdown(
                """
                **Interpretation:** Bivariate analysis helps compare fraudulent and non-fraudulent groups.
                Differences in variables such as income, employment duration, and housing type can provide useful signals, although the separation may not be strong for any single variable.
                """
            )

            st.subheader("Multivariate Analysis")
            heatmap_fig, corr_df = plot_heatmap(eda_df, "Correlation Heatmap")
            st.pyplot(heatmap_fig)
            plt.close(heatmap_fig)

            if "TARGET" in corr_df.columns:
                st.subheader("Correlation with Target")
                target_corr = corr_df["TARGET"].sort_values(ascending=False).reset_index()
                target_corr.columns = ["Feature", "Correlation with TARGET"]
                st.dataframe(make_unique_columns(target_corr), use_container_width=True, hide_index=True)

            st.markdown(
                """
                **Interpretation:** Multivariate analysis shows that no single numerical feature has a very strong linear correlation with the target, which suggests that fraud detection depends on more complex combinations of variables.
                This justifies the use of machine-learning models such as Logistic Regression, ANN, and XGBoost after careful preprocessing.
                """
            )

st.markdown("---")
st.markdown("This app combines client-level fraud prediction with a detailed EDA report covering data understanding, preprocessing, and visual analysis.")