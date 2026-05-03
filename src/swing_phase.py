"""
スイングフェーズ推定とクラブヘッド軌跡補完。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np


class SwingPhase(str, Enum):
    ADDRESS = "Address"
    BACKSWING = "Backswing"
    DOWNSWING = "Downswing"
    IMPACT = "Impact"
    FOLLOW_THROUGH = "Follow-through"
    FINISH = "Finish"


@dataclass
class SwingPhaseResult:
    phase_by_frame: list[SwingPhase]
    head_points: list[Optional[np.ndarray]]
    start_idx: int
    top_idx: int
    impact_idx: int
    finish_idx: int


def _landmark_xy(landmarks: Dict, idx: int, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    lm = landmarks.get(idx)
    if lm is None:
        return None
    if lm.get("visibility", 1.0) < 0.5:
        return None
    return np.array([float(lm["x"]) * frame_w, float(lm["y"]) * frame_h], dtype=np.float32)


def _hand_center(keypoints: Optional[Dict], frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    landmarks = keypoints.get("landmarks", {})
    lw = _landmark_xy(landmarks, 15, frame_w, frame_h)
    rw = _landmark_xy(landmarks, 16, frame_w, frame_h)
    if lw is not None and rw is not None:
        return (lw + rw) * 0.5
    return lw if lw is not None else rw


def _shoulder_center(keypoints: Optional[Dict], frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    landmarks = keypoints.get("landmarks", {})
    ls = _landmark_xy(landmarks, 11, frame_w, frame_h)
    rs = _landmark_xy(landmarks, 12, frame_w, frame_h)
    if ls is not None and rs is not None:
        return (ls + rs) * 0.5
    return ls if ls is not None else rs


def extract_club_head_point(
    sample_points: Optional[np.ndarray],
    keypoints: Optional[Dict],
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    """クラブ軸サンプル点からクラブヘッド位置（先端）を返す。"""
    if sample_points is None or len(sample_points) < 2:
        return None

    p0 = sample_points[0].astype(np.float32)
    p1 = sample_points[-1].astype(np.float32)
    ref = _hand_center(keypoints, frame_w, frame_h)
    if ref is None:
        ref = _shoulder_center(keypoints, frame_w, frame_h)
    if ref is None:
        ref = np.array([frame_w * 0.5, frame_h * 0.5], dtype=np.float32)

    d0 = float(np.linalg.norm(p0 - ref))
    d1 = float(np.linalg.norm(p1 - ref))
    return p0 if d0 >= d1 else p1


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def is_segment_likely_truncated(
    mask: Optional[np.ndarray],
    sample_points: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    *,
    edge_margin_px: int = 2,
) -> bool:
    """クラブセグメントが画角端で切れている可能性を判定する。"""
    if mask is None or mask.size == 0:
        return False

    touches_border = bool(
        np.any(mask[0, :] > 0)
        or np.any(mask[-1, :] > 0)
        or np.any(mask[:, 0] > 0)
        or np.any(mask[:, -1] > 0)
    )
    if not touches_border:
        return False

    if sample_points is None or len(sample_points) == 0:
        return True

    m = int(max(0, edge_margin_px))
    for p in (sample_points[0], sample_points[-1]):
        x, y = int(p[0]), int(p[1])
        if x <= m or x >= frame_w - 1 - m or y <= m or y >= frame_h - 1 - m:
            return True
    return touches_border


def _fill_nan_linear(values: np.ndarray, *, max_gap: int = 24) -> np.ndarray:
    out = values.astype(np.float32).copy()
    n = len(out)
    valid = np.where(~np.isnan(out))[0]
    if len(valid) == 0:
        return out

    first = int(valid[0])
    if first <= 3:
        out[:first] = out[first]
    last = int(valid[-1])
    if (n - 1 - last) <= 3:
        out[last + 1 :] = out[last]

    for li, ri in zip(valid[:-1], valid[1:]):
        gap = int(ri - li - 1)
        if gap <= 0 or gap > max_gap:
            continue
        lv = float(out[li])
        rv = float(out[ri])
        for k in range(1, gap + 1):
            t = k / float(gap + 1)
            out[li + k] = (1.0 - t) * lv + t * rv
    return out


def _wrist_y_series(
    keypoints_seq: list[Optional[Dict]],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    ys = np.full((len(keypoints_seq),), np.nan, dtype=np.float32)
    for i, kp in enumerate(keypoints_seq):
        hand = _hand_center(kp, frame_w, frame_h)
        if hand is not None:
            ys[i] = float(hand[1])
    return _fill_nan_linear(ys, max_gap=24)


def _estimate_start_from_head_motion(
    head_points: list[Optional[np.ndarray]],
    frame_w: int,
    frame_h: int,
    *,
    fallback_idx: int = 0,
) -> int:
    """
    Address -> Backswing 開始点をクラブヘッド移動開始で推定する。
    できるだけ早めに反応するため、低め閾値 + 短い連続移動で判定する。
    """
    n = len(head_points)
    if n < 3:
        return max(0, min(fallback_idx, n - 1))

    head_interp = interpolate_head_points(
        head_points,
        max_gap=8,
        max_edge_hold=2,
    )
    valid_idx = [i for i, p in enumerate(head_interp) if p is not None]
    if len(valid_idx) < 3:
        return max(0, min(fallback_idx, n - 1))

    pts = np.vstack([head_interp[i] for i in valid_idx]).astype(np.float32)
    ref_count = min(4, len(pts))
    ref = np.mean(pts[:ref_count], axis=0)
    dist = np.linalg.norm(pts - ref, axis=1)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    speed_s = _smooth(speed, window=5)

    diag = float(np.hypot(frame_w, frame_h))
    dist_thr = max(2.0, diag * 0.006)
    speed_thr = max(0.8, diag * 0.0018)

    start_local = 0
    for i in range(1, len(pts)):
        moved = dist[i] >= dist_thr
        moving_now = (i - 1) < len(speed_s) and speed_s[i - 1] >= speed_thr
        moving_next = i < len(speed_s) and speed_s[i] >= speed_thr
        if moved and (moving_now or moving_next):
            start_local = i
            break

    start_idx = int(valid_idx[start_local])
    return max(0, min(start_idx, n - 1))


def _run_segments(states: np.ndarray) -> list[tuple[int, int, int]]:
    if len(states) == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    s = 0
    v = int(states[0])
    for i in range(1, len(states)):
        if int(states[i]) != v:
            runs.append((s, i - 1, v))
            s = i
            v = int(states[i])
    runs.append((s, len(states) - 1, v))
    return runs


def _suppress_short_runs(states: np.ndarray, value: int, min_len: int, replace_with: int) -> None:
    for s, e, v in _run_segments(states):
        if v == value and (e - s + 1) < min_len:
            states[s : e + 1] = replace_with


def _macro_motion_states(y_smooth: np.ndarray) -> np.ndarray:
    """
    1次差分からマクロな上下動を抽出する。
    状態: -1=上昇(画面座標y減少), 0=停滞, +1=下降(画面座標y増加)
    """
    if len(y_smooth) < 3:
        return np.zeros((max(0, len(y_smooth) - 1),), dtype=np.int8)

    dy = np.diff(y_smooth)
    dy_s = _smooth(dy, window=9)
    abs_dy = np.abs(dy_s)
    nz = abs_dy[abs_dy > 1e-4]
    if len(nz) == 0:
        return np.zeros_like(dy_s, dtype=np.int8)

    thr = float(np.percentile(nz, 35))
    thr = max(thr, float(np.std(dy_s) * 0.25), 0.05)

    states = np.zeros_like(dy_s, dtype=np.int8)
    states[dy_s > thr] = 1
    states[dy_s < -thr] = -1

    # 微小ランの抑制でミクロな上下を停滞へ吸収
    _suppress_short_runs(states, -1, min_len=5, replace_with=0)
    _suppress_short_runs(states, 1, min_len=5, replace_with=0)
    _suppress_short_runs(states, 0, min_len=3, replace_with=0)
    return states


def _find_macro_transition_midpoint(
    states: np.ndarray,
    *,
    start_idx: int,
    from_state: int,
    to_state: int,
) -> Optional[int]:
    """
    マクロ遷移 from -> flat(0) -> to を探し、flat 区間の中間点を返す。
    flat が無い場合は境界中点を返す。
    返り値は y_smooth のフレームindex。
    """
    runs = _run_segments(states)
    if not runs:
        return None

    for i, (s, e, v) in enumerate(runs):
        if v != from_state or e < start_idx:
            continue
        j = i + 1
        flat_s: Optional[int] = None
        flat_e: Optional[int] = None
        while j < len(runs) and runs[j][2] == 0:
            if flat_s is None:
                flat_s = runs[j][0]
            flat_e = runs[j][1]
            j += 1

        if j < len(runs) and runs[j][2] == to_state:
            if flat_s is not None and flat_e is not None:
                mid = (flat_s + flat_e) // 2
            else:
                mid = (e + runs[j][0]) // 2
            return int(np.clip(mid + 1, 0, len(states)))
    return None


def interpolate_head_points(
    head_points: list[Optional[np.ndarray]],
    *,
    max_gap: int = 12,
    max_edge_hold: int = 5,
    can_interpolate_frame: Optional[list[bool]] = None,
) -> list[Optional[np.ndarray]]:
    """欠損ヘッド点を短区間のみ補間する。"""
    n = len(head_points)
    if can_interpolate_frame is None or len(can_interpolate_frame) != n:
        can_interpolate = [True] * n
    else:
        can_interpolate = can_interpolate_frame

    out: list[Optional[np.ndarray]] = [None] * n
    valid_indices = [i for i, p in enumerate(head_points) if p is not None]
    for i in valid_indices:
        out[i] = head_points[i]
    if not valid_indices:
        return out

    first_valid = valid_indices[0]
    if first_valid <= max_edge_hold and all(can_interpolate[:first_valid]):
        for i in range(first_valid):
            out[i] = head_points[first_valid]

    last_valid = valid_indices[-1]
    if (n - 1 - last_valid) <= max_edge_hold and all(can_interpolate[last_valid + 1 :]):
        for i in range(last_valid + 1, n):
            out[i] = head_points[last_valid]

    for li, ri in zip(valid_indices[:-1], valid_indices[1:]):
        gap = ri - li - 1
        if gap <= 0 or gap > max_gap:
            continue
        if not all(can_interpolate[li + 1 : ri]):
            continue
        lp = head_points[li]
        rp = head_points[ri]
        if lp is None or rp is None:
            continue
        for k in range(1, gap + 1):
            t = k / float(gap + 1)
            out[li + k] = (1.0 - t) * lp + t * rp

    return out


def estimate_swing_phases(
    head_points: list[Optional[np.ndarray]],
    keypoints_seq: list[Optional[Dict]],
    frame_w: int,
    frame_h: int,
    *,
    can_interpolate_frame: Optional[list[bool]] = None,
) -> SwingPhaseResult:
    """クラブヘッド軌跡と姿勢情報からスイングフェーズを推定する。"""
    n = len(head_points)
    if n == 0:
        return SwingPhaseResult([], [], 0, 0, 0, 0)

    interpolated = interpolate_head_points(
        head_points,
        can_interpolate_frame=can_interpolate_frame,
    )
    wrist_y = _wrist_y_series(keypoints_seq, frame_w, frame_h)
    valid_wrist = np.where(~np.isnan(wrist_y))[0]

    # 手首がほぼ取れない場合のみヘッド軌跡の y で代替
    if len(valid_wrist) < max(5, n // 10):
        for i, p in enumerate(interpolated):
            if p is not None:
                wrist_y[i] = float(p[1])
        wrist_y = _fill_nan_linear(wrist_y, max_gap=16)
        valid_wrist = np.where(~np.isnan(wrist_y))[0]

    if len(valid_wrist) < 3:
        return SwingPhaseResult(
            phase_by_frame=[SwingPhase.ADDRESS for _ in range(n)],
            head_points=interpolated,
            start_idx=0,
            top_idx=max(0, n // 3),
            impact_idx=max(0, (2 * n) // 3),
            finish_idx=max(0, int(0.85 * n)),
        )

    y_filled = wrist_y.copy()
    y_filled[np.isnan(y_filled)] = np.nanmean(y_filled)
    y_s = _smooth(y_filled, window=7)

    # Address -> Backswing は手首ではなくクラブヘッドの動き出しで判定
    start_idx = _estimate_start_from_head_motion(
        interpolated,
        frame_w,
        frame_h,
        fallback_idx=0,
    )

    macro_states = _macro_motion_states(y_s)

    # Backswing -> Downswing: 上昇(-1) -> 停滞(0) -> 下降(+1) の停滞中間
    top_idx = _find_macro_transition_midpoint(
        macro_states,
        start_idx=start_idx + 1,
        from_state=-1,
        to_state=1,
    )
    if top_idx is None:
        top_idx = max(start_idx + 1, int(n * 0.35))

    # Impact: 下降(+1) -> 停滞(0) -> 上昇(-1) の停滞中間（最下点周辺）
    impact_idx = _find_macro_transition_midpoint(
        macro_states,
        start_idx=top_idx + 1,
        from_state=1,
        to_state=-1,
    )
    if impact_idx is None:
        impact_idx = min(n - 1, max(top_idx + 2, int(n * 0.6)))

    # Follow-through -> Finish: 上昇(-1) -> 停滞(0) -> 下降(+1) の停滞中間 + 少し後
    finish_peak = _find_macro_transition_midpoint(
        macro_states,
        start_idx=impact_idx + 1,
        from_state=-1,
        to_state=1,
    )
    if finish_peak is None:
        finish_peak = min(n - 1, max(impact_idx + 4, int(n * 0.82)))
    finish_idx = min(n - 1, finish_peak + max(2, int(round(0.02 * n))))

    if top_idx < start_idx:
        top_idx = start_idx
    if impact_idx <= top_idx:
        impact_idx = min(n - 1, top_idx + 1)
    if finish_idx <= impact_idx:
        finish_idx = min(n - 1, max(impact_idx + 1, int(0.85 * n)))

    phases = [SwingPhase.ADDRESS for _ in range(n)]
    for i in range(n):
        if i < start_idx:
            phases[i] = SwingPhase.ADDRESS
        elif i <= top_idx:
            phases[i] = SwingPhase.BACKSWING
        elif i < max(impact_idx - 1, top_idx + 1):
            phases[i] = SwingPhase.DOWNSWING
        elif i <= min(n - 1, impact_idx + 1):
            phases[i] = SwingPhase.IMPACT
        elif i < finish_idx:
            phases[i] = SwingPhase.FOLLOW_THROUGH
        else:
            phases[i] = SwingPhase.FINISH

    return SwingPhaseResult(
        phase_by_frame=phases,
        head_points=interpolated,
        start_idx=start_idx,
        top_idx=top_idx,
        impact_idx=impact_idx,
        finish_idx=finish_idx,
    )
