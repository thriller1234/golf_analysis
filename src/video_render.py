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
from src.swing_phase import (
    SwingPhase,
    SwingPhaseResult,
    estimate_swing_phases,
    extract_club_head_point,
    is_segment_likely_truncated,
)
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


def _analyze_frames(
    frames: list[np.ndarray],
    estimator: PoseEstimator,
    club_segmentor: Optional["ClubSegmentor"],
    *,
    total_out_frames: int,
) -> tuple[
    list[Optional[dict]],
    list[Optional[np.ndarray]],
    list[np.ndarray],
    Optional[SwingPhaseResult],
]:
    keypoints_seq: list[Optional[dict]] = []
    masks: list[Optional[np.ndarray]] = []
    club_pts: list[np.ndarray] = []
    head_raw: list[Optional[np.ndarray]] = []
    can_interp: list[bool] = []

    if not frames:
        return keypoints_seq, masks, club_pts, None

    h, w = frames[0].shape[:2]
    for i, frame in enumerate(frames):
        timestamp_ms = int(round(i * 1000.0 * OUTPUT_DURATION_SEC / max(total_out_frames, 1)))
        mask: Optional[np.ndarray] = None
        cpts = np.empty((0, 2), dtype=np.int32)
        if club_segmentor is not None:
            mask, cpts = club_segmentor.segment_frame(frame, num_samples=_CLUB_MASK_SAMPLES)
        kp = estimator.process_frame_video(frame, timestamp_ms=timestamp_ms)

        masks.append(mask)
        club_pts.append(cpts)
        keypoints_seq.append(kp)
        clipped = is_segment_likely_truncated(mask, cpts, w, h)
        can_interp.append(not clipped)
        head_raw.append(None if clipped else extract_club_head_point(cpts, kp, w, h))

    phase_result = None
    if club_segmentor is not None:
        phase_result = estimate_swing_phases(
            head_raw,
            keypoints_seq,
            w,
            h,
            can_interpolate_frame=can_interp,
        )
    return keypoints_seq, masks, club_pts, phase_result


def _draw_head_trace(
    frame: np.ndarray,
    head_points: list[Optional[np.ndarray]],
    phase_by_frame: list[SwingPhase],
    current_idx: int,
) -> np.ndarray:
    if not head_points or not phase_by_frame:
        return frame

    out = frame.copy()
    overlay = out.copy()
    h, w = out.shape[:2]

    color_ab = (215, 215, 90)   # Address + Backswing
    color_ds = (210, 135, 230)  # Downswing
    max_i = min(current_idx, len(head_points) - 1, len(phase_by_frame) - 1)
    for i in range(max_i):
        p1 = head_points[i]
        p2 = head_points[i + 1]
        if p1 is None or p2 is None:
            continue
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        # 補完点が画角外の場合は無理に描画しない
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        phase = phase_by_frame[i]
        color: Optional[tuple[int, int, int]] = None
        if phase in (SwingPhase.ADDRESS, SwingPhase.BACKSWING):
            color = color_ab
        elif phase == SwingPhase.DOWNSWING:
            color = color_ds
        if color is None:
            continue

        cv2.line(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0)


def _classify_swing_plane(
    head_points: list[Optional[np.ndarray]],
    phase_by_frame: list[SwingPhase],
    frame_w: int,
    frame_h: int,
) -> str:
    """
    1スイング全体で Backswing と Downswing の軌跡位置関係を評価し、
    Shallow / Steep / On-plane を返す。
    """
    bs: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    n = min(len(head_points), len(phase_by_frame))
    for i in range(n):
        p = head_points[i]
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x < frame_w and 0.0 <= y < frame_h):
            continue
        ph = phase_by_frame[i]
        if ph in (SwingPhase.ADDRESS, SwingPhase.BACKSWING):
            bs.append(np.array([x, y], dtype=np.float32))
        elif ph == SwingPhase.DOWNSWING:
            ds.append(np.array([x, y], dtype=np.float32))

    if len(bs) < 5 or len(ds) < 5:
        return "On-plane"

    # y帯ごとに x を比較して、軌道全体の左右差を集計
    ys_all = np.array([p[1] for p in bs + ds], dtype=np.float32)
    y_min, y_max = float(np.min(ys_all)), float(np.max(ys_all))
    if y_max <= y_min + 1e-3:
        return "On-plane"

    n_bins = 14
    bins = np.linspace(y_min, y_max, n_bins + 1)
    deltas: list[float] = []
    bs_arr = np.vstack(bs)
    ds_arr = np.vstack(ds)
    for bi in range(n_bins):
        yl, yr = float(bins[bi]), float(bins[bi + 1])
        in_bs = (bs_arr[:, 1] >= yl) & (bs_arr[:, 1] < yr)
        in_ds = (ds_arr[:, 1] >= yl) & (ds_arr[:, 1] < yr)
        if int(np.sum(in_bs)) < 1 or int(np.sum(in_ds)) < 1:
            continue
        x_bs = float(np.median(bs_arr[in_bs, 0]))
        x_ds = float(np.median(ds_arr[in_ds, 0]))
        deltas.append(x_ds - x_bs)

    if len(deltas) < 3:
        return "On-plane"

    delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    thr = max(8.0, frame_w * 0.03)  # 「大きく」左/右にある場合のみ
    if delta <= -thr:
        return "Shallow"
    if delta >= thr:
        return "Steep"
    return "On-plane"


def _draw_plane_label(frame: np.ndarray, plane_label: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        f"Plane: {plane_label}",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return out


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
    frames = _read_all_frames(input_path)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"読み込めるフレームがありません: {input_path}")

    out_fps = n_frames / OUTPUT_DURATION_SEC
    height, width = frames[0].shape[:2]
    size = (width, height)

    kp_seq, mask_seq, club_pts_seq, phase_result = _analyze_frames(
        frames,
        estimator,
        club_segmentor,
        total_out_frames=n_frames,
    )
    plane_label: Optional[str] = None
    if phase_result is not None and phase_result.phase_by_frame:
        plane_label = _classify_swing_plane(
            phase_result.head_points,
            phase_result.phase_by_frame,
            width,
            height,
        )

    writer = _open_writer(output_path, out_fps, size)
    try:
        for frame_idx, frame in enumerate(frames):
            out = frame.copy()
            if club_segmentor is not None:
                out = draw_club_overlay(out, mask_seq[frame_idx], club_pts_seq[frame_idx])

            kp = kp_seq[frame_idx]
            if kp is not None:
                out = visualizer.visualize_keypoints(out, kp)

            if phase_result is not None and phase_result.phase_by_frame:
                out = _draw_head_trace(
                    out,
                    phase_result.head_points,
                    phase_result.phase_by_frame,
                    frame_idx,
                )
                if (
                    plane_label is not None
                    and phase_result.phase_by_frame[frame_idx] == SwingPhase.DOWNSWING
                ):
                    out = _draw_plane_label(out, plane_label)
                phase_text = f"Phase: {phase_result.phase_by_frame[frame_idx].value}"
                phase_pos = (12, 72) if label else (12, 36)
                cv2.putText(
                    out,
                    phase_text,
                    phase_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

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
    finally:
        writer.release()


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

    km_seq, mm_seq, cm_seq, phase_my = _analyze_frames(
        frames_my,
        estimator_my,
        club_segmentor,
        total_out_frames=f_out,
    )
    kp_seq, pm_seq, cp_seq, phase_pro = _analyze_frames(
        frames_pro,
        estimator_pro,
        club_segmentor,
        total_out_frames=f_out,
    )
    plane_my: Optional[str] = None
    plane_pro: Optional[str] = None
    if phase_my is not None and phase_my.phase_by_frame:
        h1, w1 = frames_my[0].shape[:2]
        plane_my = _classify_swing_plane(phase_my.head_points, phase_my.phase_by_frame, w1, h1)
    if phase_pro is not None and phase_pro.phase_by_frame:
        h2, w2 = frames_pro[0].shape[:2]
        plane_pro = _classify_swing_plane(phase_pro.head_points, phase_pro.phase_by_frame, w2, h2)

    writer: Optional[cv2.VideoWriter] = None
    try:
        for j in range(f_out):
            idx_my = min(j * n1 // f_out, n1 - 1)
            idx_pro = min(j * n2 // f_out, n2 - 1)
            frame_my = frames_my[idx_my]
            frame_pro = frames_pro[idx_pro]

            om = frame_my.copy()
            op_ = frame_pro.copy()
            if club_segmentor is not None:
                om = draw_club_overlay(om, mm_seq[idx_my], cm_seq[idx_my])
                op_ = draw_club_overlay(op_, pm_seq[idx_pro], cp_seq[idx_pro])

            km = km_seq[idx_my]
            kp = kp_seq[idx_pro]

            if km is not None:
                om = visualizer.visualize_keypoints(om, km)
            if kp is not None:
                op_ = visualizer.visualize_keypoints(op_, kp)

            if phase_my is not None and phase_my.phase_by_frame:
                om = _draw_head_trace(om, phase_my.head_points, phase_my.phase_by_frame, idx_my)
                if plane_my is not None and phase_my.phase_by_frame[idx_my] == SwingPhase.DOWNSWING:
                    om = _draw_plane_label(om, plane_my)
                cv2.putText(
                    om,
                    f"Phase: {phase_my.phase_by_frame[idx_my].value}",
                    (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
            if phase_pro is not None and phase_pro.phase_by_frame:
                op_ = _draw_head_trace(op_, phase_pro.head_points, phase_pro.phase_by_frame, idx_pro)
                if plane_pro is not None and phase_pro.phase_by_frame[idx_pro] == SwingPhase.DOWNSWING:
                    op_ = _draw_plane_label(op_, plane_pro)
                cv2.putText(
                    op_,
                    f"Phase: {phase_pro.phase_by_frame[idx_pro].value}",
                    (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
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
                writer = _open_writer(output_path, out_fps, (w, h))

            writer.write(combined)

        if writer is None:
            raise ValueError("比較動画を書き出せませんでした（読み込めるフレームがありません）。")
    finally:
        if writer is not None:
            writer.release()
