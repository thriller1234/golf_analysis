"""
二値マスクからシャフト方向の主軸に沿って点を等間隔サンプルする。
"""

from __future__ import annotations

import numpy as np


def sample_points_from_mask(mask: np.ndarray, n: int = 8) -> np.ndarray:
    """
    Args:
        mask: 単チャネル、0/255 近傍の uint8。クラブ領域が非ゼロ。
        n: サンプル点数（両端含む）

    Returns:
        shape (n, 2) の int32、画像座標 (x, y)。有効画素が少なすぎる場合は (0, 2)。
    """
    if mask is None or n < 2:
        return np.empty((0, 2), dtype=np.int32)

    ys, xs = np.where(mask > 127)
    if len(xs) < 5:
        return np.empty((0, 2), dtype=np.int32)

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    t = centered @ direction
    t_min, t_max = float(t.min()), float(t.max())
    ts = np.linspace(t_min, t_max, n)
    sampled = mean + np.outer(ts, direction)
    sampled = np.round(sampled)
    sampled[:, 0] = np.clip(sampled[:, 0], 0, mask.shape[1] - 1)
    sampled[:, 1] = np.clip(sampled[:, 1], 0, mask.shape[0] - 1)
    return sampled.astype(np.int32)
