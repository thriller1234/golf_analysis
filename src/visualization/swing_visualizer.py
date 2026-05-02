"""
動画フレームへの骨格オーバーレイ描画（OpenCV のみ）
"""

from typing import Dict, Optional

import cv2
import numpy as np

# MediaPipe Pose の骨格接続（簡易）
POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


class SwingVisualizer:
    """フレームへの骨格描画"""

    def visualize_keypoints(
        self,
        frame: np.ndarray,
        keypoints: Optional[Dict],
        draw_connections: bool = True,
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        if keypoints is None:
            return annotated_frame
        landmarks = keypoints.get("landmarks", {})
        if not landmarks:
            return annotated_frame

        h, w = frame.shape[:2]
        joint_points = {}
        for idx, landmark in landmarks.items():
            x = int(landmark["x"] * w)
            y = int(landmark["y"] * h)
            visibility = landmark.get("visibility", 1.0)
            if visibility < 0.5:
                continue
            joint_points[idx] = (x, y)
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

        if draw_connections:
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx in joint_points and end_idx in joint_points:
                    pt1 = joint_points[start_idx]
                    pt2 = joint_points[end_idx]
                    cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)

        return annotated_frame
