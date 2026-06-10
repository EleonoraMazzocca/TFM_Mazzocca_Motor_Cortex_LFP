"""Shared metrics for cVAE evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


def compute_mmd(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = 1.0) -> float:
    """Maximum Mean Discrepancy with an RBF kernel between two sample arrays.

    Pass bandwidth=None to use a median heuristic computed from X only
    (self-distances excluded). The default keeps historical behavior.
    """
    if bandwidth is None:
        dists = euclidean_distances(X, X)
        np.fill_diagonal(dists, np.nan)
        bandwidth = float(np.nanmedian(dists))
        bandwidth = max(bandwidth, 1e-6)
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    xx = rbf_kernel(X, X, gamma=gamma).mean()
    yy = rbf_kernel(Y, Y, gamma=gamma).mean()
    xy = rbf_kernel(X, Y, gamma=gamma).mean()
    return float(xx + yy - 2.0 * xy)
