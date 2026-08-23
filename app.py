"""
Module 7 — Engine Remaining Useful Life (RUL) Predictor Dashboard
====================================================================
An interactive Streamlit app that loads the trained XGBoost model from
Module 5/6 and lets a user pick any test engine to see its sensor
history and predicted Remaining Useful Life, with a color-coded
maintenance recommendation and a transparency note about the model's
known mid-life bias (discovered in Module 6).

REQUIRED FILES (must sit in the SAME FOLDER as this app.py):
  - xgb_model.pkl          (Module 5, Step 5.6 / 5.7 — the saved model)
  - feature_cols.json      (Module 4/5 — the exact feature column list,
                             in the exact order the model was trained on)
  - test_engine_data.csv   (Module 4 — cleaned, feature-engineered data
                             for the held-out test engines, e.g. 81-100)

RUN WITH (from a real terminal, in this same folder):
  streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit command in the script
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Engine RUL Predictor",
    page_icon="\u2708\ufe0f",
    layout="centered",
)

MODEL_PATH = "xgb_model.pkl"
FEATURES_PATH = "feature_cols.json"
DATA_PATH = "test_engine_data.csv"

# Thresholds for the status zones — tune these to taste, but keep them
# consistent with what you write in your README/essay.
RED_THRESHOLD = 25      # <= this many predicted cycles -> Critical
YELLOW_THRESHOLD = 75   # <= this many predicted cycles -> Monitor
                         # above YELLOW_THRESHOLD -> Healthy

# The zone where Module 6's directional-bias analysis found the model's
# largest late-bias (i.e. the model tends to OVERestimate remaining life
# the most in this window). Update these numbers if you rerun Module 6
# with a different model or dataset split.
BIAS_ZONE_LOW = 51
BIAS_ZONE_HIGH = 75
BIAS_ZONE_MAGNITUDE = 13.76  # mean signed error (cycles) found in that zone


# ---------------------------------------------------------------------
# STEP 7.1 — Load saved artifacts (cached so this only runs once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [p for p in (MODEL_PATH, FEATURES_PATH, DATA_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    data = pd.read_csv(DATA_PATH)
    return model, features, data, []


model, feature_cols, test_data, missing_files = load_artifacts()

st.title("\u2708\ufe0f Engine Remaining Useful Life Predictor")
st.caption("AI-driven Condition-Based Maintenance \u2014 NASA C-MAPSS (FD001) \u2014 XGBoost model")

# ---------------------------------------------------------------------
# Friendly error screen instead of a raw crash if files aren't found
# ---------------------------------------------------------------------
if missing_files:
    st.error("This app can't find the files it needs to run.")
    st.markdown(
        "The following file(s) are missing from the folder this app is "
        "running in:"
    )
    for m in missing_files:
        st.code(m)
    st.markdown(
        "**Fix:** copy `xgb_model.pkl`, `feature_cols.json`, and "
        "`test_engine_data.csv` (exported from your notebook in Module 4/5) "
        "into the **same folder** as `app.py`, then refresh this page.\n\n"
        f"This app's current working folder is:\n`{os.getcwd()}`"
    )
    st.stop()

# ---------------------------------------------------------------------
# Validate that the CSV actually contains every column the model expects
# ---------------------------------------------------------------------
missing_cols = [c for c in feature_cols if c not in test_data.columns]
if missing_cols:
    st.error("`test_engine_data.csv` is missing columns the model expects.")
    st.write("Missing columns:", missing_cols)
    st.markdown(
        "This usually means the CSV was exported from a different pipeline "
        "run than the one that trained the model. Re-export "
        "`test_engine_data.csv` from the same notebook session that saved "
        "`feature_cols.json`."
    )
    st.stop()

if "unit_number" not in test_data.columns or "time_in_cycles" not in test_data.columns:
    st.error(
        "`test_engine_data.csv` must include `unit_number` and "
        "`time_in_cycles` columns to power the engine selector and sensor plot."
    )
    st.stop()


# ---------------------------------------------------------------------
# STEP 7.2 — Engine selector
# ---------------------------------------------------------------------
engine_ids = sorted(test_data["unit_number"].unique())
selected_engine = st.selectbox("Select an engine to inspect:", engine_ids)

engine_df = test_data[test_data["unit_number"] == selected_engine].copy()
engine_df = engine_df.sort_values("time_in_cycles")

min_cycle = int(engine_df["time_in_cycles"].min())
max_cycle = int(engine_df["time_in_cycles"].max())

st.markdown(
    f"Engine **#{selected_engine}** has **{len(engine_df)}** recorded flight "
    f"cycles (from cycle {min_cycle} to cycle {max_cycle}) in this test set."
)

# ---------------------------------------------------------------------
# STEP 7.2b — Cycle slider: inspect the engine at ANY point in its life,
# not just its final recorded cycle. Without this, every engine's most
# recent row is its failure point (since this data runs to failure), so
# every prediction would always land in the CRITICAL zone regardless of
# which engine is selected.
# ---------------------------------------------------------------------
if min_cycle == max_cycle:
    selected_cycle = min_cycle
    st.caption(f"Only one recorded cycle available for this engine (cycle {min_cycle}).")
else:
    selected_cycle = st.slider(
        "Inspect engine at flight cycle:",
        min_value=min_cycle,
        max_value=max_cycle,
        value=max_cycle,
        help="Drag left to see this engine earlier in its life (healthier), "
             "or right to see it closer to its recorded end-of-life point.",
    )

# Only show sensor history UP TO the selected cycle, so the chart reflects
# "what we'd have known at this point in time" rather than always showing
# the engine's entire life story end-to-end.
visible_df = engine_df[engine_df["time_in_cycles"] <= selected_cycle]


# ---------------------------------------------------------------------
# STEP 7.3 — Sensor trend plot for the selected engine, up to the
# currently selected cycle
# ---------------------------------------------------------------------
st.subheader(f"Sensor History \u2014 Engine {selected_engine} (up to cycle {selected_cycle})")

# Pick the first available rolling-average sensor column to plot, so this
# doesn't hard-code a sensor name that might not exist in your real data.
roll_cols = [c for c in feature_cols if c.endswith("_roll5")]
plot_col = roll_cols[0] if roll_cols else feature_cols[0]

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(visible_df["time_in_cycles"], visible_df[plot_col], color="crimson", linewidth=1.5)
ax.set_xlim(min_cycle, max_cycle)  # keep the x-axis stable while scrubbing
ax.set_xlabel("Flight Cycle")
ax.set_ylabel(plot_col)
ax.set_title(f"{plot_col} \u2014 Engine {selected_engine}, cycles {min_cycle}\u2013{selected_cycle}")
ax.grid(alpha=0.3)
st.pyplot(fig)


# ---------------------------------------------------------------------
# STEP 7.4 — Run the prediction at the SELECTED cycle, not always the
# engine's last recorded row
# ---------------------------------------------------------------------
last_cycle_row = engine_df[engine_df["time_in_cycles"] == selected_cycle]
current_cycle = selected_cycle
features_to_predict = last_cycle_row[feature_cols]

predicted_rul = model.predict(features_to_predict)[0]
predicted_rul = max(0, int(round(predicted_rul)))

st.markdown("---")
st.subheader("Current Status")
st.metric(
    label=f"Predicted Remaining Useful Life (as of cycle {current_cycle})",
    value=f"{predicted_rul} cycles",
)


# ---------------------------------------------------------------------
# STEP 7.5 — Color-coded status indicator
# ---------------------------------------------------------------------
st.subheader("Maintenance Recommendation")

if predicted_rul <= RED_THRESHOLD:
    st.error(
        f"**CRITICAL \u2014 Action Required.** Engine has an estimated "
        f"{predicted_rul} cycles remaining. Ground for physical inspection."
    )
    st.caption(
        f"Reasoning: {RED_THRESHOLD} cycles is the redline threshold used "
        f"in this dashboard \u2014 at this stage, failure is imminent and "
        f"maintenance must be scheduled immediately."
    )
elif predicted_rul <= YELLOW_THRESHOLD:
    st.warning(
        f"**MONITOR \u2014 Maintenance Window Approaching.** Estimated "
        f"{predicted_rul} cycles remaining. Schedule inspection soon."
    )
else:
    st.success(
        f"**HEALTHY \u2014 Routine Operations.** Estimated {predicted_rul} "
        f"cycles remaining, a safe buffer above the monitoring threshold."
    )

# ---------------------------------------------------------------------
# STEP 7.6 — Surface the Module 6 mid-life bias finding, when relevant
# ---------------------------------------------------------------------
if BIAS_ZONE_LOW <= predicted_rul <= BIAS_ZONE_HIGH:
    st.info(
        f"\U0001F4CA **Model transparency note:** predictions in the "
        f"{BIAS_ZONE_LOW}\u2013{BIAS_ZONE_HIGH} cycle range showed this "
        f"model's largest late-bias during evaluation (+{BIAS_ZONE_MAGNITUDE} "
        f"cycles on average \u2014 see Module 6). Treat this estimate as "
        f"slightly optimistic, not exact."
    )

st.markdown("---")
st.caption(
    "Built on NASA's C-MAPSS (FD001) dataset. Model: XGBoost Regressor, "
    "evaluated with both standard error metrics (RMSE/MAE) and NASA's "
    "asymmetric scoring function (Saxena et al., PHM08)."
)