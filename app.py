import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import kagglehub

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    file_path = f"{path}/insurance.csv"
    return pd.read_csv(file_path)


# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    return None


# -----------------------------
# Feature Engineering for Demo
# -----------------------------
def prepare_input_dataframe(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )
    return input_df


def demo_predict(age, sex, bmi, children, smoker, region):
    """
    Fallback demo prediction when no pkl model is available.
    This is only for UI/demo purposes.
    """
    base = 2000
    age_factor = age * 220
    bmi_factor = max(0, bmi - 21) * 260
    children_factor = children * 450
    smoker_factor = 24000 if smoker == "yes" else 0
    sex_factor = 300 if sex == "male" else 0

    region_map = {
        "northeast": 1200,
        "northwest": 800,
        "southeast": 1500,
        "southwest": 900,
    }
    region_factor = region_map.get(region, 0)

    prediction = (
        base
        + age_factor
        + bmi_factor
        + children_factor
        + smoker_factor
        + sex_factor
        + region_factor
    )
    return round(prediction, 2)


def safe_model_predict(model, input_df):
    """
    Tries to use the loaded model safely.
    Works best if your pipeline already includes preprocessing.
    """
    try:
        pred = model.predict(input_df)
        return float(pred[0])
    except Exception:
        return None


# -----------------------------
# Load Resources
# -----------------------------
data = load_data()
model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("Insurance Charge Predictor")
st.markdown(
    """
    A demo web application for exploring the insurance dataset and predicting estimated medical charges.
    The app can run in two modes:

    - **Model Mode**: uses `models/model.pkl` if available
    - **Demo Mode**: uses a realistic formula-based estimate until the trained model is added
    """
)

if model is not None:
    st.success("Prediction model detected. The app is currently using `models/model.pkl`.")
else:
    st.info("No `model.pkl` found yet. The app is running in demo mode with a simulated predictor.")

# -----------------------------
# Sidebar Filters for EDA
# -----------------------------
st.sidebar.header("Dataset Filters")

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sorted(data["sex"].unique()),
    default=sorted(data["sex"].unique())
)

selected_smoker = st.sidebar.multiselect(
    "Smoker",
    options=sorted(data["smoker"].unique()),
    default=sorted(data["smoker"].unique())
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=sorted(data["region"].unique()),
    default=sorted(data["region"].unique())
)

filtered_data = data[
    (data["sex"].isin(selected_sex)) &
    (data["smoker"].isin(selected_smoker)) &
    (data["region"].isin(selected_region))
]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Predict Charges", "About Model"])

# =============================
# TAB 1: Dashboard
# =============================
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(filtered_data):,}")
    col2.metric("Average Charges", f"${filtered_data['charges'].mean():,.2f}")
    col3.metric("Average BMI", f"{filtered_data['bmi'].mean():.2f}")
    col4.metric("Average Age", f"{filtered_data['age'].mean():.2f}")

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_data, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered_data.describe(), use_container_width=True)

    st.subheader("Visual Analytics")

    chart_type = st.selectbox(
        "Select Visualization",
        [
            "Age Histogram",
            "BMI Histogram",
            "Charges Histogram",
            "Charges by Smoker",
            "Charges by Region",
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "Age Histogram":
        ax.hist(filtered_data["age"], bins=20)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")

    elif chart_type == "BMI Histogram":
        ax.hist(filtered_data["bmi"], bins=20)
        ax.set_title("BMI Distribution")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Count")

    elif chart_type == "Charges Histogram":
        ax.hist(filtered_data["charges"], bins=20)
        ax.set_title("Charges Distribution")
        ax.set_xlabel("Charges")
        ax.set_ylabel("Count")

    elif chart_type == "Charges by Smoker":
        filtered_data.boxplot(column="charges", by="smoker", ax=ax)
        ax.set_title("Charges by Smoker")
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Charges")
        plt.suptitle("")

    elif chart_type == "Charges by Region":
        filtered_data.boxplot(column="charges", by="region", ax=ax)
        ax.set_title("Charges by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Charges")
        plt.suptitle("")

    st.pyplot(fig)

    st.subheader("Grouped Average Charges")
    grouped = (
        filtered_data.groupby(["region", "smoker"])["charges"]
        .mean()
        .reset_index()
        .rename(columns={"charges": "avg_charges"})
    )
    grouped["avg_charges"] = grouped["avg_charges"].round(2)
    st.dataframe(grouped, use_container_width=True)

# =============================
# TAB 2: Prediction
# =============================
with tab2:
    st.subheader("Predict Medical Insurance Charges")

    st.markdown("Enter customer details below to generate a predicted insurance charge.")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        age = st.slider("Age", min_value=18, max_value=64, value=30)
        bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=27.5, step=0.1)
        children = st.slider("Number of Children", min_value=0, max_value=5, value=0)

    with right_col:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    input_df = prepare_input_dataframe(age, sex, bmi, children, smoker, region)

    st.markdown("### Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Charges", use_container_width=True):
        prediction = None

        if model is not None:
            prediction = safe_model_predict(model, input_df)

        if prediction is None:
            prediction = demo_predict(age, sex, bmi, children, smoker, region)
            mode_label = "Demo Prediction"
        else:
            mode_label = "Model Prediction"

        st.markdown("### Prediction Result")
        st.success(f"{mode_label}: **${prediction:,.2f}**")

        st.markdown("### Interpretation")
        if prediction < 8000:
            st.write("This customer appears to be in a relatively lower-cost insurance segment.")
        elif prediction < 20000:
            st.write("This customer appears to be in a moderate-cost insurance segment.")
        else:
            st.write("This customer appears to be in a high-cost insurance segment, potentially influenced by smoking status, BMI, or age.")

# =============================
# TAB 3: About Model
# =============================
with tab3:
    st.subheader("Model Integration Guide")

    st.markdown(
        """
        This application is ready for a trained `.pkl` model.

        **Expected path**
        - `models/model.pkl`

        **Recommended approach**
        - Train a scikit-learn pipeline
        - Include preprocessing inside the pipeline
        - Save the full pipeline as `model.pkl`

        **Why this is best**
        - The app can directly accept raw inputs like `sex`, `smoker`, and `region`
        - You do not need to manually encode features in the Streamlit app
        """
    )

    st.code(
        """# Example training idea
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

X = data.drop("charges", axis=1)
y = data["charges"]

cat_cols = ["sex", "smoker", "region"]
num_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)

pipeline.fit(X, y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
""",
        language="python"
    )

    if model is not None:
        st.success("A model is currently loaded and available for predictions.")
    else:
        st.warning("No model file is available yet. Add `models/model.pkl` to enable real predictions.")