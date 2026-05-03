"""
入力動画に骨格＋（任意で）クラブセグのオーバーレイをかけて mp4 を出力する。

出力の実時間は常に OUTPUT_DURATION_SEC 秒。入力の全フレームを使い、再生速度を調整する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from src.pose_estimation import PoseEstimator
from src.visualization import SwingVisualizer
from src.visualization.club_overlay import draw_club_overlay

if TYPE_CHECKING:
    from src.club_detection import ClubSegmentor

# クラブマスク主軸上のサンプル点数（CLI では指定しない）
_CLUB_MASK_SAMPLES = 8

# 書き出し動画の長さ（秒）。元動画の全フレームをこの長さに収める
OUTPUT_DURATION_SEC = 10.0


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


def _count_frames_by_scan(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {path}")
    n = 0
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            n += 1
    finally:
        cap.release()
    return n


def _read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def render_pose_overlay_video(
    input_path: Path,
    output_path: Path,
    estimator: PoseEstimator,
    visualizer: SwingVisualizer,
    *,
    label: Optional[str] = None,
    club_segmentor: Optional["ClubSegmentor"] = None,
) -> None:
    """1 本の動画: 姿勢推定 +（任意）YOLO クラブセグ。全フレームを使い OUTPUT_DURATION_SEC 秒に伸縮して書き出す。"""
    n_frames = _count_frames_by_scan(input_path)
    if n_frames == 0:
        raise ValueError(f"読み込めるフレームがありません: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {input_path}")

    out_fps = n_frames / OUTPUT_DURATION_SEC
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    writer = _open_writer(output_path, out_fps, size)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int(round(frame_idx * 1000.0 * OUTPUT_DURATION_SEC / n_frames))

            out = frame.copy()
            if club_segmentor is not None:
                mask, cpts = club_segmentor.segment_frame(
                    frame, num_samples=_CLUB_MASK_SAMPLES
                )
                out = draw_club_overlay(out, mask, cpts)

            kp = estimator.process_frame_video(frame, timestamp_ms=timestamp_ms)
            if kp is not None:
                out = visualizer.visualize_keypoints(out, kp)

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
    club_segmentor: Optional["ClubSegmentor"] = None,
) -> None:
    """
    2 本の動画を横並びに。各動画の全フレームを使い、それぞれ OUTPUT_DURATION_SEC 秒に伸縮して同期して書き出す。
    """
    frames_my = _read_all_frames(my_path)
    frames_pro = _read_all_frames(pro_path)
    n1 = len(frames_my)
    n2 = len(frames_pro)
    if n1 == 0 or n2 == 0:
        raise ValueError("比較動画を書き出せませんでした（読み込めるフレームがありません）。")

    f_out = max(n1, n2)
    out_fps = f_out / OUTPUT_DURATION_SEC

    writer: Optional[cv2.VideoWriter] = None

    try:
        for j in range(f_out):
            idx_my = min(j * n1 // f_out, n1 - 1)
            idx_pro = min(j * n2 // f_out, n2 - 1)
            frame_my = frames_my[idx_my]
            frame_pro = frames_pro[idx_pro]

            timestamp_ms = int(round(j * 1000.0 * OUTPUT_DURATION_SEC / f_out))

            om = frame_my.copy()
            op_ = frame_pro.copy()
            if club_segmentor is not None:
                mm, cm = club_segmentor.segment_frame(
                    frame_my, num_samples=_CLUB_MASK_SAMPLES
                )
                om = draw_club_overlay(om, mm, cm)
                pm, cp = club_segmentor.segment_frame(
                    frame_pro, num_samples=_CLUB_MASK_SAMPLES
                )
                op_ = draw_club_overlay(op_, pm, cp)

            km = estimator_my.process_frame_video(frame_my, timestamp_ms=timestamp_ms)
            kp = estimator_pro.process_frame_video(frame_pro, timestamp_ms=timestamp_ms)

            if km is not None:
                om = visualizer.visualize_keypoints(om, km)
            if kp is not None:
                op_ = visualizer.visualize_keypoints(op_, kp)

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
                writer = _open_writer(output_path, out_fps, (w, h))

            writer.write(combined)

        if writer is None:
            raise ValueError("比較動画を書き出せませんでした（読み込めるフレームがありません）。")
    finally:
        if writer is not None:
            writer.release()
