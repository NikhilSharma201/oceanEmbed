"""
anomaly.py — Simple marine-heatwave-style anomaly flagging.

Approach: compare a predicted temperature at a depth against a regional
climatological baseline (mean + std for that region/depth); flag as
anomalous if it deviates beyond a z-score threshold. This mirrors how
real marine heatwave detection works (e.g. Hobday et al. definitions use
a similar deviation-from-climatology idea), simplified for this prototype.
"""

import numpy as np


def compute_climatology(subsurface_temp, region_id):
    """Returns per-region, per-depth (mean, std) baseline."""
    n_regions = region_id.max() + 1
    n_depths = subsurface_temp.shape[1]
    means = np.zeros((n_regions, n_depths))
    stds = np.zeros((n_regions, n_depths))
    for r in range(n_regions):
        mask = region_id == r
        means[r] = subsurface_temp[mask].mean(axis=0)
        stds[r] = subsurface_temp[mask].std(axis=0)
    return means, stds


def flag_anomalies(pred_temp, region_id, means, stds, z_threshold=2.0):
    """
    pred_temp: (n, n_depths) predicted (median) temperatures
    Returns: (n, n_depths) boolean array, True where anomalous (marine-heatwave-like)
    """
    baseline_mean = means[region_id]
    baseline_std = stds[region_id]
    z_scores = (pred_temp - baseline_mean) / (baseline_std + 1e-6)
    return z_scores > z_threshold, z_scores
