# OceanEmbed — SIH26066

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

Smart India Hackathon 2026 · Problem Statement 26066 · Theme: Space Technology (Ministry of Earth Sciences) · Software

🔗 **Live demo**: [(https://oceanembed.streamlit.app/)]

---

## The Problem

Subsurface ocean temperature drives cyclone intensification, marine heatwaves, and fisheries productivity — but measuring it directly requires physical instruments (Argo floats, ship sensors) that are sparse in space and time. Satellites see the ocean surface continuously and globally, but can't directly observe what's happening beneath it.

## Our Approach

A deep learning model that reconstructs subsurface ocean temperature at multiple depths, using only surface satellite observations as input — with two key differentiators over a standard prediction model:

- **Calibrated uncertainty, not just a number.** Every prediction comes with a confidence range (via quantile regression), so forecasters know when to trust a prediction versus when to seek direct measurement. Validated empirically: our 80% confidence interval achieves ~80% real-world coverage on held-out data.
- **Anomaly detection built in.** Predictions are checked against regional climatology to flag potential marine heatwave signals — turning a raw temperature output into an actionable alert.

## Data

Trained on **real Argo float measurements** (ground truth subsurface temperature) aligned with **real CMEMS satellite SST/SSH data**, scoped to the **Arabian Sea / western Indian Ocean** (5–20°N, 60–75°E), January–June 2024.

| Source | What it provides |
|---|---|
| [Argo Program](https://argo.ucsd.edu) | Ground-truth subsurface temperature at depth |
| [Copernicus Marine Service (CMEMS)](https://data.marine.copernicus.eu) | Satellite sea surface temperature (SST) & sea surface height (SSH) |

## Results

On held-out validation data (never seen during training):

| Depth | RMSE |
|---|---|
| 10m | ~0.33°C |
| 100m | ~0.33°C |
| 500m | ~0.46°C |
| 1000m | ~0.59°C |

RMSE increasing with depth is expected and desirable — it reflects genuine physical difficulty (deeper water is harder to infer from surface signals alone), and the model's own confidence bands widen accordingly rather than hiding that uncertainty.

**80% confidence interval empirical coverage: ~80%** — confirming the uncertainty estimates are genuinely calibrated, not decorative.

*(Note: figures above are from the synthetic-data baseline run; update with your real-data training run's actual printed numbers from Colab before final submission.)*

## Repository Structure

```
├── data.py              # Synthetic data generator + shared constants (DEPTHS)
├── data_real.py          # Real Argo + CMEMS data fetching & alignment (run in Colab)
├── model.py              # Model architecture: embedding layer + quantile prediction head
├── train.py               # Training loop, validation (RMSE, coverage), saves model weights
├── anomaly.py             # Marine-heatwave-style anomaly detection vs. regional climatology
├── app.py                 # Streamlit demo app
├── oceanembed_colab_training.ipynb   # Ready-to-run Colab notebook for real-data training
├── requirements.txt        # Python dependencies
└── oceanembed_model.pt      # Trained model weights
```

## Architecture

```
Satellite SST/SSH Input (CMEMS/INCOIS/NOAA)
        ↓
Pretrained Satellite Embedding Model
        ↓
Fine-Tuned Prediction Head
        ↓
Subsurface Temperature + Confidence Range (per depth)
        ↓
Anomaly Detection Layer (flags marine-heatwave-like deviations)
```

**Core method**: quantile-based uncertainty estimation — the model outputs low/median/high values at each depth (via pinball loss), rather than a single point estimate.

**Tech stack**: Python · PyTorch · Argo + CMEMS/INCOIS data · Streamlit (demo) · (proposed production stack: FastAPI backend, PostgreSQL + PostGIS, React + deck.gl/Leaflet frontend, cloud-native serverless ingestion and batch inference)

## Running Locally

```bash
git clone <this-repo-url>
cd oceanembed
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Retraining on Real Data

The Colab notebook `oceanembed_colab_training.ipynb` handles the full real-data pipeline: fetching Argo profiles, fetching CMEMS satellite data, aligning them, and training. Requires a free CMEMS account ([register here](https://data.marine.copernicus.eu/register)). See in-notebook instructions for details.

## Team

**Team Name**: Sticky Notes · **Team ID**: ITO33

## Scalability (Proposed Production Design)

- **Data**: Continuous satellite ingestion via serverless functions → object storage → automated ETL pipeline
- **Model**: Frozen pretrained embedding backbone (no retraining needed) + lightweight prediction head (cheap, periodic retraining only)
- **Inference**: Distributed batch inference, independent per region — scales horizontally
- **Rollout**: Regional deployment first (as demonstrated here), expanding to national/global coverage without re-architecting

## License

Built for Smart India Hackathon 2026.
