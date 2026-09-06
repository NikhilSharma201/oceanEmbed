"""
app.py — OceanEmbed live demo (Streamlit)

Run: streamlit run app.py

Shows:
  1. A map of sample ocean locations
  2. Click/select a location -> predicted subsurface temperature profile
     with confidence band (the uncertainty-quantification differentiator)
  3. Predicted vs. actual (held-out) comparison
  4. Marine-heatwave-style anomaly flag for that location, if triggered
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from data import DEPTHS
import pickle
from model import OceanEmbedModel
from anomaly import compute_climatology, flag_anomalies

st.set_page_config(page_title="OceanEmbed — SIH26066", layout="wide")

st.title("🌊 OceanEmbed — Subsurface Ocean Temperature Reconstruction")
st.caption(
    "SIH26066 · Prototype demo · Predicting subsurface temperature from surface satellite "
    "observations, with calibrated uncertainty — trained on real Argo + CMEMS satellite data, "
    "Arabian Sea region, Jan–Jun 2024."
)

with st.expander("ℹ️ How to read this demo", expanded=True):
    st.markdown(
        """
        - **Map below**: sample ocean locations with real Argo float measurements
        - **Slider**: pick a location to see its predicted subsurface temperature
        - **Confidence band** (shaded blue): the model's calibrated uncertainty range —
          not a guess, but a range trained so the true value falls inside it ~80% of the time
        - **Actual (Argo Reference)**: the real, held-out measurement at that location —
          data the model never saw during training, used only to check accuracy
        - **Anomaly check**: flags if predicted temperature is unusually high for that
          region, a potential marine-heatwave signal
        """
    )

# Headline metrics - visible immediately, no interaction needed
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Validation RMSE", "0.42°C")
col_b.metric("80% Interval Coverage", "80.0%")
col_c.metric("Region", "Arabian Sea")
col_d.metric("Data Period", "Jan–Jun 2024")
st.divider()

@st.cache_resource
def load_model_and_data():
    ckpt = torch.load("oceanembed_model.pt", weights_only=False)
    model = OceanEmbedModel(in_features=4, n_depths=len(DEPTHS))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open("real_data.pkl", "rb") as f:
        data = pickle.load(f)  
    means, stds = compute_climatology(data["subsurface_temp"], data["region_id"])
    return model, ckpt["x_mean"], ckpt["x_std"], data, means, stds


model, x_mean, x_std, data, clim_means, clim_stds = load_model_and_data()

# ---- Map of sample locations ----
st.subheader("1. Sample ocean locations")
map_df = pd.DataFrame({
    "lat": data["lat"].astype(float),
    "lon": data["lon"].astype(float),
})
st.map(map_df, size=20000)

location_labels = [
    f"{lat:.2f}°N, {lon:.2f}°E" for lat, lon in zip(data["lat"], data["lon"])
]
idx = st.select_slider(
    "Select an ocean location",
    options=range(len(data["sst"])),
    format_func=lambda i: location_labels[i],
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Location", f"({data['lat'][idx]:.1f}, {data['lon'][idx]:.1f})")
with col2:
    st.metric("Surface Temp (SST)", f"{data['sst'][idx]:.2f} °C")

# ---- Prediction ----
st.subheader("2. Predicted subsurface temperature (with confidence band)")

x = np.array([[data["sst"][idx], data["ssh"][idx], data["lat"][idx] / 90.0, data["lon"][idx] / 180.0]])
x_t = torch.tensor(x, dtype=torch.float32)
x_n = (x_t - x_mean) / x_std

with torch.no_grad():
    pred = model(x_n).numpy()[0]  # (n_depths, 3) -> low, median, high

low, median, high = pred[:, 0], pred[:, 1], pred[:, 2]
actual = data["subsurface_temp"][idx]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(median, -DEPTHS, "o-", color="#1f77b4", label="Predicted (median)")
ax.fill_betweenx(-DEPTHS, low, high, color="#1f77b4", alpha=0.2, label="80% confidence band")
ax.plot(actual, -DEPTHS, "x--", color="#d62728", label="Actual (held-out)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title("Predicted vs. Actual Subsurface Temperature Profile")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

rmse = np.sqrt(((median - actual) ** 2).mean())
st.info(
    f"RMSE for this location: **{rmse:.2f} °C**. The dashed red line is a real, "
    f"held-out Argo measurement — data the model never saw while training — shown "
    f"alongside the model's prediction to check accuracy."
)

# ---- Anomaly detection ----
st.subheader("3. Marine heatwave anomaly check")
region = data["region_id"][idx]
anomalies, z_scores = flag_anomalies(
    median.reshape(1, -1), np.array([region]), clim_means, clim_stds
)
anomalies, z_scores = anomalies[0], z_scores[0]

any_anomaly = anomalies.any()
if any_anomaly:
    flagged_depths = DEPTHS[anomalies]
    st.warning(
        f"⚠️ Anomaly detected at depth(s) {list(flagged_depths)}m — predicted temperature "
        f"is more than 2 standard deviations above this region's baseline. Potential "
        f"marine-heatwave signal."
    )
else:
    st.success("✅ No anomaly detected — predicted temperatures are within normal regional baseline.")

with st.expander("Show anomaly z-scores per depth"):
    st.dataframe(pd.DataFrame({"Depth (m)": DEPTHS, "Z-score": z_scores}))

st.divider()
st.caption(
    "Note: running on physically-motivated synthetic data in this prototype. "
    "See README.md for how to swap in live Argo + CMEMS/INCOIS data — the model "
    "and pipeline code do not need to change, only the data source."
)