"""
Streamlit front-end.

Priority for predictions:
  1. If registry/champion.json + models/model.pkl are present → use the model.
  2. Else → fall back to the demo-mode formula.

Run locally:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.utils import (
    demo_predict,
    load_dataset,
    load_model,
    load_params,
    model_path,
    project_root,
)

# --------------------------------------------------------------------
# Optional New Relic instrumentation
# --------------------------------------------------------------------
NR_ENABLED = False
try:
    import newrelic.agent  # type: ignore

    if os.environ.get("NEW_RELIC_LICENSE_KEY"):
        cfg = os.environ.get("NEW_RELIC_CONFIG_FILE", "newrelic.ini")
        if Path(cfg).exists():
            try:
                newrelic.agent.initialize(cfg)
            except Exception:
                pass
        NR_ENABLED = True
except ImportError:
    pass


def nr_event(name: str, payload: dict) -> None:
    """Record a custom event in New Relic; no-op when NR isn't configured."""
    if not NR_ENABLED:
        return
    try:
        newrelic.agent.record_custom_event(name, payload)
    except Exception:
        pass


# --------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------
PARAMS = load_params()
st.set_page_config(
    page_title=PARAMS["app"]["title"],
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset(PARAMS)


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model(PARAMS)


@st.cache_data(show_spinner=False)
def cached_champion() -> dict | None:
    champ_file = project_root() / "registry" / "champion.json"
    if not champ_file.exists():
        return None
    with open(champ_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.title(PARAMS["app"]["title"])
st.markdown(
    f"{PARAMS['app']['description']} The app can run in two modes:\n\n"
    "- **Model Mode**: uses `models/model.pkl` if available\n"
    "- **Demo Mode**: uses a realistic formula-based estimate until the trained model is added"
)

model = cached_model()
champ = cached_champion()

if champ and model is not None:
    st.success(
        f"🏆 Champion model loaded · "
        f"R²={champ['metrics']['r2']:.3f} · "
        f"MAE=${champ['metrics']['mae']:,.0f} · "
        f"RMSE=${champ['metrics']['rmse']:,.0f} · "
        f"registered {champ['registered_at']}"
    )
elif model is not None:
    st.success(f"Loaded trained model from `{model_path(PARAMS).relative_to(project_root())}`.")
else:
    st.info("No `model.pkl` found yet. The app is running in demo mode with a simulated predictor.")

# Lightweight ping event so we can confirm telemetry is flowing on every page load
nr_event("AppPageView", {"has_model": model is not None, "has_champion": champ is not None})

# --------------------------------------------------------------------
# Sidebar — dataset filters
# --------------------------------------------------------------------
df_full = cached_dataset()

st.sidebar.header("Dataset Filters")
sex_options = PARAMS["features"]["sex"]
smoker_options = PARAMS["features"]["smoker"]
region_options = PARAMS["features"]["region"]

sex_sel = st.sidebar.multiselect("Sex", sex_options, default=sex_options)
smoker_sel = st.sidebar.multiselect("Smoker", smoker_options, default=smoker_options)
region_sel = st.sidebar.multiselect("Region", region_options, default=region_options)

df = df_full[
    df_full["sex"].isin(sex_sel)
    & df_full["smoker"].isin(smoker_sel)
    & df_full["region"].isin(region_sel)
    ].reset_index(drop=True)

# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab_dashboard, tab_predict, tab_about = st.tabs(["Dashboard", "Predict Charges", "About Model"])

# -------- Dashboard --------
with tab_dashboard:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Average Charges", f"${df['charges'].mean():,.2f}" if len(df) else "—")
    c3.metric("Average BMI", f"{df['bmi'].mean():.2f}" if len(df) else "—")
    c4.metric("Average Age", f"{df['age'].mean():.2f}" if len(df) else "—")

    st.header("Filtered Dataset")
    st.dataframe(df, use_container_width=True, height=360)

    st.header("Summary Statistics")
    if len(df):
        st.dataframe(df[["age", "bmi", "children", "charges"]].describe(), use_container_width=True)
    else:
        st.write("No rows match the current filters.")

    st.header("Visual Analytics")
    viz_choice = st.selectbox(
        "Select Visualization",
        [
            "Age Histogram",
            "BMI Histogram",
            "Charges Histogram",
            "Charges vs Age (scatter)",
            "Charges by Smoker (box)",
        ],
    )

    if len(df):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if viz_choice == "Age Histogram":
            ax.hist(df["age"], bins=15)
            ax.set_title("Age Distribution")
            ax.set_xlabel("Age");
            ax.set_ylabel("Count")
        elif viz_choice == "BMI Histogram":
            ax.hist(df["bmi"], bins=20)
            ax.set_title("BMI Distribution")
            ax.set_xlabel("BMI");
            ax.set_ylabel("Count")
        elif viz_choice == "Charges Histogram":
            ax.hist(df["charges"], bins=25)
            ax.set_title("Charges Distribution")
            ax.set_xlabel("Charges ($)");
            ax.set_ylabel("Count")
        elif viz_choice == "Charges vs Age (scatter)":
            colors = df["smoker"].map({"yes": "tab:red", "no": "tab:blue"})
            ax.scatter(df["age"], df["charges"], c=colors, alpha=0.6, s=18)
            ax.set_title("Charges vs Age (red = smoker)")
            ax.set_xlabel("Age");
            ax.set_ylabel("Charges ($)")
        elif viz_choice == "Charges by Smoker (box)":
            data = [df.loc[df["smoker"] == s, "charges"] for s in smoker_options]
            ax.boxplot(data, labels=smoker_options)
            ax.set_title("Charges by Smoker Status")
            ax.set_ylabel("Charges ($)")
        st.pyplot(fig, clear_figure=True)

    st.header("Grouped Average Charges")
    if len(df):
        grouped = (
            df.groupby(["region", "smoker"])["charges"]
            .mean()
            .round(2)
            .reset_index(name="avg_charges")
        )
        st.dataframe(grouped, use_container_width=True)

# -------- Predict Charges --------
with tab_predict:
    st.header("Predict Medical Insurance Charges")
    st.write("Enter customer details below to generate a predicted insurance charge.")

    fcfg = PARAMS["features"]
    col_l, col_r = st.columns(2)

    with col_l:
        age = st.slider("Age", fcfg["age"]["min"], fcfg["age"]["max"], fcfg["age"]["default"])
        bmi = st.slider(
            "BMI",
            float(fcfg["bmi"]["min"]),
            float(fcfg["bmi"]["max"]),
            float(fcfg["bmi"]["default"]),
            step=float(fcfg["bmi"].get("step", 0.1)),
        )
        children = st.slider(
            "Number of Children",
            fcfg["children"]["min"],
            fcfg["children"]["max"],
            fcfg["children"]["default"],
        )

    with col_r:
        sex = st.selectbox("Sex", fcfg["sex"], index=fcfg["sex"].index("male"))
        smoker = st.selectbox("Smoker", fcfg["smoker"], index=fcfg["smoker"].index("no"))
        region = st.selectbox("Region", fcfg["region"], index=0)

    input_row = pd.DataFrame(
        [{"age": age, "sex": sex, "bmi": bmi, "children": children,
          "smoker": smoker, "region": region}]
    )

    st.subheader("Input Summary")
    st.dataframe(input_row, use_container_width=True)

    if st.button("Predict Charges", use_container_width=True, type="primary"):
        t0 = time.time()
        prediction_mode = "demo"
        pred = None

        if model is not None:
            try:
                pred = float(model.predict(input_row)[0])
                prediction_mode = "champion" if champ else "model"
                st.success(f"Predicted charge ({prediction_mode}): **${pred:,.2f}**")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                pred = demo_predict(input_row.iloc[0].to_dict())
                prediction_mode = "demo_fallback"
                st.warning(f"Fell back to demo prediction: **${pred:,.2f}**")
        else:
            pred = demo_predict(input_row.iloc[0].to_dict())
            st.success(f"Predicted charge (demo mode): **${pred:,.2f}**")

        latency_ms = (time.time() - t0) * 1000.0
        nr_event("InsurancePrediction", {
            "age": int(age),
            "bmi": float(bmi),
            "children": int(children),
            "sex": str(sex),
            "smoker": str(smoker),
            "region": str(region),
            "predicted_charge": float(pred) if pred is not None else 0.0,
            "mode": prediction_mode,
            "latency_ms": latency_ms,
        })

# -------- About Model --------
with tab_about:
    st.header("Model Registry")

    if champ:
        st.subheader("Current Champion")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("R²", f"{champ['metrics']['r2']:.4f}")
        cc2.metric("MAE", f"${champ['metrics']['mae']:,.0f}")
        cc3.metric("RMSE", f"${champ['metrics']['rmse']:,.0f}")

        st.caption(
            f"Model type: `{champ['model_type']}` · "
            f"Artifact: `{champ['model_path']}` · "
            f"Registered: `{champ['registered_at']}` · "
            f"Version: `{champ['version']}`"
        )

        with st.expander("Raw champion.json"):
            st.json(champ)
    else:
        st.info("No champion registered yet. CI registers one on every push to `main`.")

    st.header("Observability")
    if NR_ENABLED:
        st.success(
            "New Relic agent is active. Custom events `InsurancePrediction` and `AppPageView` are being recorded.")
    else:
        st.info("New Relic is not configured. Set `NEW_RELIC_LICENSE_KEY` to enable telemetry.")

    st.header("Model Integration Guide")
    st.write("This application is ready for a trained `.pkl` model.")

    st.subheader("Expected path")
    st.markdown("- `models/model.pkl`")
