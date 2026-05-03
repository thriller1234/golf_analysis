"""
クラブマスクと主軸サンプル点のオーバーレイ描画。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def draw_club_overlay(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    points: np.ndarray,
    *,
    mask_alpha: float = 0.35,
    mask_color: tuple[int, int, int] = (255, 160, 0),
    line_color: tuple[int, int, int] = (0, 200, 255),
    point_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    mask: HxW uint8、None または空なら骨格のみと同様に元フレームを返す相当。
    points: (n,2) int32 (x,y)
    """
    out = frame.copy()
    if mask is not None and mask.size > 0 and mask.sum() > 0:
        m = (mask > 127).astype(np.float32)
        overlay = out.astype(np.float32)
        for c in range(3):
            overlay[..., c] = overlay[..., c] * (1.0 - mask_alpha * m) + float(
                mask_color[c]
            ) * (mask_alpha * m)
        out = np.clip(overlay, 0, 255).astype(np.uint8)

    if points is not None and len(points) >= 1:
        fh, fw = frame.shape[:2]
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < fw and 0 <= y < fh:
                cv2.circle(out, (x, y), 5, point_color, -1, cv2.LINE_AA)
        for i in range(len(points) - 1):
            p1 = (int(points[i][0]), int(points[i][1]))
            p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.line(out, p1, p2, line_color, 2, cv2.LINE_AA)

    return out
