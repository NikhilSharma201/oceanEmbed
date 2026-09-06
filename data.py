"""
data.py — Data layer for OceanEmbed (SIH26066)

Two data sources are provided:

1. SYNTHETIC (used by default, works fully offline):
   A physically-motivated generator that mimics the real relationship
   between surface satellite observations (SST, SSH) and subsurface
   temperature at depth. It is NOT real data — it's built so the whole
   pipeline (model, training, uncertainty, anomaly detection, viz) can be
   built and demoed before real data is plugged in.

   The physics it encodes (all standard oceanography, used only to make
   the synthetic data realistic, not invented):
   - Temperature decreases with depth (thermocline structure)
   - Warmer/higher surface (SST/SSH) correlates with a deeper, warmer
     thermocline (real physical relationship satellites exploit)
   - Random regional + seasonal noise, so it isn't a trivial linear fit

2. REAL DATA (for you to run once you have live internet access):
   - `load_argo_real()`   -> requires `pip install argopy`
   - `load_cmems_real()`  -> requires a free CMEMS account + `copernicusmarine` package
   These are fully written and documented below. They are not run in
   this sandbox because it has no network access to ERDDAP/CMEMS servers.
   Swap SYNTHETIC for REAL by calling these instead of `generate_synthetic_dataset`.
"""

import numpy as np

DEPTHS = np.array([10, 50, 100, 200, 500, 1000])  # meters
N_REGIONS = 6  # a handful of ocean regions, not global — keeps scope realistic


def generate_synthetic_dataset(n_samples=4000, seed=42):
    """
    Returns:
        sst: (n_samples,) sea surface temperature (deg C)
        ssh: (n_samples,) sea surface height anomaly (m)
        lat, lon: (n_samples,) coordinates (used for map viz + region grouping)
        subsurface_temp: (n_samples, len(DEPTHS)) "true" temperature at each depth
    """
    rng = np.random.default_rng(seed)

    # Assign each sample to one of a few regions with different baseline climates
    region_id = rng.integers(0, N_REGIONS, size=n_samples)
    region_base_temp = np.array([28, 26, 22, 18, 24, 15])[region_id]  # deg C, varies by region
    region_lat = np.array([5, 15, 30, 45, -10, -30])[region_id]

    lat = region_lat + rng.normal(0, 3, n_samples)
    lon = rng.uniform(-180, 180, n_samples)

    # Surface signals
    sst = region_base_temp + rng.normal(0, 1.2, n_samples)
    # SSH anomaly correlates with subsurface heat content (real relationship: warm eddies raise SSH)
    ssh = 0.15 * (sst - region_base_temp.mean()) + rng.normal(0, 0.08, n_samples)

    # Subsurface temperature: starts at SST, decays toward deep-ocean baseline (~4C) with depth.
    # SSH anomaly nudges the decay profile (a warm eddy -> deeper/slower decay), a real physical effect,
    # but only as a secondary correction — not double-counted with the SST->depth relationship itself.
    subsurface_temp = np.zeros((n_samples, len(DEPTHS)))
    deep_ocean_temp = 4.0
    for i, d in enumerate(DEPTHS):
        decay = np.exp(-d / 300.0)  # thermocline-like exponential decay, 1.0 at surface -> ~0 at depth
        ssh_correction = ssh * 3.0 * decay  # small secondary nudge from SSH anomaly, fades with depth
        noise = rng.normal(0, 0.3 + 0.3 * (d / 1000), n_samples)  # more noise at depth (harder to infer)
        subsurface_temp[:, i] = deep_ocean_temp + (sst - deep_ocean_temp) * decay + ssh_correction + noise

    return {
        "sst": sst.astype(np.float32),
        "ssh": ssh.astype(np.float32),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "region_id": region_id,
        "subsurface_temp": subsurface_temp.astype(np.float32),
        "depths": DEPTHS,
    }


def load_argo_real(bbox=None, start_date="2023-01-01", end_date="2023-12-31"):
    """
    REAL DATA LOADER — requires internet + `pip install argopy`.
    Not called in this sandbox (no network access to Argo's ERDDAP servers).

    Example use on your own machine:
        from argopy import DataFetcher
        fetcher = DataFetcher()
        if bbox is None:
            bbox = [-180, 180, -90, 90, 0, 1000]  # lon_min, lon_max, lat_min, lat_max, depth_min, depth_max
        ds = fetcher.region(bbox + [start_date, end_date]).load().data
        df = ds.to_dataframe().reset_index()
        return df  # columns include LATITUDE, LONGITUDE, PRES (depth), TEMP, TIME
    """
    raise NotImplementedError(
        "Run this on a machine with internet access. "
        "pip install argopy, then uncomment the implementation in this docstring."
    )


def load_cmems_real(product="SST_GLO_PHY_L4", bbox=None, start_date=None, end_date=None):
    """
    REAL DATA LOADER — requires a free CMEMS account + `pip install copernicusmarine`.
    Not called in this sandbox (no network access to CMEMS servers).

    Example use on your own machine:
        import copernicusmarine
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_obs-sst_glo_phy_l4",
            minimum_longitude=bbox[0], maximum_longitude=bbox[1],
            minimum_latitude=bbox[2], maximum_latitude=bbox[3],
            start_datetime=start_date, end_datetime=end_date,
        )
        return ds.to_dataframe().reset_index()  # includes SST, and for SSH use a SEALEVEL product
    """
    raise NotImplementedError(
        "Run this on a machine with internet access and CMEMS credentials. "
        "pip install copernicusmarine, then uncomment the implementation in this docstring."
    )


if __name__ == "__main__":
    d = generate_synthetic_dataset(200)
    print("Synthetic dataset sample:")
    print("SST range:", d["sst"].min(), d["sst"].max())
    print("Subsurface temp at depths", DEPTHS, ":")
    print(d["subsurface_temp"][:3])
