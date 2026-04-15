from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Risk Predictor", page_icon="💳", layout="wide")


# =========================================================
# PATH HELPERS
# =========================================================
def find_project_root(start: Path) -> Path:
    """Find project root by looking for expected data/report folders."""
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

LOG_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
ANN_MODEL_PATH = MODEL_DIR / "ann_model.pkl"
SCALER_PATH = SPLITS_DIR / "minmax_scaler.pkl"
X_TRAIN_PATH = SPLITS_DIR / "X_train.csv"
MODEL_BASE_PATH = PROCESSED_DIR / "Model_Ready_Base_Credit_Card_Dataset.csv"


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
    """Scale only the columns the saved scaler expects, keeping all columns float."""
    model_input = encoded_client.copy().astype(float)

    if not hasattr(scaler, "feature_names_in_"):
        return model_input

    scaler_columns = list(scaler.feature_names_in_)
    scaler_columns = [col for col in scaler_columns if col in model_input.columns]

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
# APP
# =========================================================
st.title("💳 Credit Card Fraud Risk Prediction App")
st.caption(
    "Insert a new client profile and predict the probability of becoming fraudulent using your trained models."
)

try:
    logistic_model, ann_model, scaler, encoded_train_columns = load_models_and_scaler()
except Exception as exc:
    st.error("Required model or preprocessing files were not found.")
    st.code(str(exc))
    st.stop()

base_df = load_base_dataset()

fallback_family_types = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
fallback_house_types = [
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
fallback_income_types = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
fallback_education_types = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]

gender_options = get_category_options(base_df, "GENDER", ["M", "F"])
family_type_options = get_category_options(base_df, "FAMILY_TYPE", fallback_family_types)
house_type_options = get_category_options(base_df, "HOUSE_TYPE", fallback_house_types)
income_type_options = get_category_options(base_df, "INCOME_TYPE", fallback_income_types)
education_type_options = get_category_options(base_df, "EDUCATION_TYPE", fallback_education_types)

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
                "Gender",
                "Age",
                "Annual Income",
                "Years Employed",
                "Begin Month",
                "Number of Children",
                "Family Size",
                "Family Type",
                "House Type",
                "Income Type",
                "Education Type",
                "Has Car",
                "Has Property",
                "Work Phone",
                "Personal Phone",
            ],
            "Value": [
                gender,
                age,
                income,
                years_employed,
                begin_month,
                no_of_child,
                family_size,
                family_type,
                house_type,
                income_type,
                education_type,
                "Yes" if has_car else "No",
                "Yes" if has_property else "No",
                "Yes" if work_phone else "No",
                "Yes" if phone else "No",
            ],
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

        if selected_model == "Logistic Regression":
            model = logistic_model
        else:
            model = ann_model

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

st.markdown("---")
st.markdown(
    "This app is designed to behave similarly to your sample Shiny app, but implemented in Streamlit for your trained credit-card fraud models."
)