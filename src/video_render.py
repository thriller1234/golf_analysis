"""
入力動画に骨格オーバーレイをかけて mp4 を出力する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.pose_estimation import PoseEstimator
from src.visualization import SwingVisualizer


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == 0:
        return frame
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _open_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps if fps and fps > 0 else 30.0
    writer = cv2.VideoWriter(str(path), fourcc, out_fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter を開けませんでした: {path}")
    return writer


def render_pose_overlay_video(
    input_path: Path,
    output_path: Path,
    estimator: PoseEstimator,
    visualizer: SwingVisualizer,
    *,
    label: Optional[str] = None,
) -> None:
    """1本の動画を読み、フレームごとに姿勢を推定して骨格を描画した mp4 を書き出す。"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    writer = _open_writer(output_path, fps, size)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            kp = estimator.process_frame_video(frame, timestamp_ms=timestamp_ms)
            if kp is not None:
                out = visualizer.visualize_keypoints(frame, kp)
            else:
                out = frame.copy()
            if label:
                cv2.putText(
                    out,
                    label,
                    (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(out)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()


def render_side_by_side_overlay(
    my_path: Path,
    pro_path: Path,
    output_path: Path,
    estimator_my: PoseEstimator,
    estimator_pro: PoseEstimator,
    visualizer: SwingVisualizer,
    *,
    panel_height: int = 720,
    label_my: str = "myswing",
    label_pro: str = "proswing",
) -> None:
    """
    2本の動画を同じフレーム番号で同期させ、短い方の長さまで横並びで1つの mp4 に出力する。
    detect_for_video はストリームごとに状態を持つため、左右で PoseEstimator を分ける。
    """
    cap_my = cv2.VideoCapture(str(my_path))
    cap_pro = cv2.VideoCapture(str(pro_path))
    if not cap_my.isOpened():
        raise ValueError(f"動画を開けません: {my_path}")
    if not cap_pro.isOpened():
        raise ValueError(f"動画を開けません: {pro_path}")

    fps_my = float(cap_my.get(cv2.CAP_PROP_FPS) or 30.0)
    fps_pro = float(cap_pro.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = min(fps_my, fps_pro)

    writer: Optional[cv2.VideoWriter] = None
    idx = 0

    try:
        while True:
            ret_a, frame_my = cap_my.read()
            ret_b, frame_pro = cap_pro.read()
            if not ret_a or not ret_b:
                break

            timestamp_ms = int(round(idx * 1000.0 / fps))
            km = estimator_my.process_frame_video(frame_my, timestamp_ms=timestamp_ms)
            kp = estimator_pro.process_frame_video(frame_pro, timestamp_ms=timestamp_ms)

            om = (
                visualizer.visualize_keypoints(frame_my, km)
                if km is not None
                else frame_my.copy()
            )
            op_ = (
                visualizer.visualize_keypoints(frame_pro, kp)
                if kp is not None
                else frame_pro.copy()
            )

            cv2.putText(
                om,
                label_my,
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                op_,
                label_pro,
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            lm = _resize_to_height(om, panel_height)
            rp = _resize_to_height(op_, panel_height)
            combined = np.hstack([lm, rp])

            if writer is None:
                h, w = combined.shape[:2]
                writer = _open_writer(output_path, fps, (w, h))

            writer.write(combined)
            idx += 1

        if writer is None:
            raise ValueError("比較動画を書き出せませんでした（読み込めるフレームがありません）。")
    finally:
        if writer is not None:
            writer.release()
        cap_my.release()
        cap_pro.release()
