"""
YOLOv8 Segment によるクラブ領域推論とマスク取得。

複数クラス（例: COCO 事前学習）のチェックポイントでは、クラブ相当クラスが判別できない場合は
オーバーレイを出さない（人物などを誤ってマスクしない）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    YOLO = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from .mask_geometry import sample_points_from_mask


def _polygon_to_mask(
    poly_xy: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Ultralytics の polygon (N, 2) をフレームサイズの二値マスクにする。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    if poly_xy is None or len(poly_xy) < 3:
        return mask
    pts = np.asarray(poly_xy, dtype=np.float32).reshape(-1, 2)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], 255)
    return mask


def _normalize_names(names_obj) -> dict[int, str]:
    """model.names を {int: str} に統一。"""
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_obj)}
    return {}


def _infer_club_class_indices(names: dict[int, str]) -> tuple[int, ...]:
    """
    クラブ用として使うクラス ID を推定。
    - 名前が golf_club / club 系 → その ID のみ
    - 単一クラスモデル → (0,) とみなす
    - COCO 等で該当なし → 空タプル（クラブとして検出しない）
    """
    club_like = []
    for idx, name in names.items():
        n = name.lower().replace("-", "_").replace(" ", "_")
        if n in ("golf_club", "club") or "golf_club" in n:
            club_like.append(idx)
    if club_like:
        return tuple(sorted(set(club_like)))
    if len(names) == 1:
        return (0,)
    return tuple()


class ClubSegmentor:
    """YOLOv8 Instance Segmentation（学習済み .pt）でクラブマスクを推論する。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | int = 0,
        max_det: int = 5,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics が必要です。pip install ultralytics を実行してください。"
            ) from _IMPORT_ERROR

        self._path = Path(model_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"クラブ用セグモデルが見つかりません: {self._path}")

        self.model = YOLO(str(self._path))
        self.device = device
        self.max_det = max_det

        names = _normalize_names(self.model.names)
        self._club_class_indices = _infer_club_class_indices(names)
        self._club_disabled = len(self._club_class_indices) == 0

        if self._club_disabled:
            print(
                "⚠️  クラブセグ: このチェックポイントには「golf_club」等のクラブ用クラス名が"
                "含まれていません（COCO 事前学習のみの可能性があります）。\n"
                "   クラブのマスク・折れ線オーバーレイはスキップします。"
                " scripts/train_yolov8_club.py でクラブのみ学習した .pt を用意してください。"
            )
        elif len(names) > 1 and self._club_class_indices:
            print(
                f"クラブセグ: 使用クラス ID = {self._club_class_indices} "
                f"({', '.join(names[i] for i in self._club_class_indices)})"
            )

    def segment_frame(
        self,
        frame: np.ndarray,
        *,
        num_samples: int = 8,
        imgsz: int | None = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        1 フレームを推論し、クラブクラスのインスタンスのうち信頼度最大のマスクとサンプル点を返す。
        """
        empty_pts = np.empty((0, 2), dtype=np.int32)
        if self._club_disabled:
            return None, empty_pts

        h, w = frame.shape[:2]
        if imgsz is None:
            m = max(h, w)
            imgsz = min(1280, max(32, int(round(m / 32)) * 32))

        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            verbose=False,
            device=self.device,
            max_det=self.max_det,
        )
        res = results[0]

        if res.masks is None or len(res.masks) == 0:
            return None, empty_pts
        if res.boxes is None or len(res.boxes) == 0:
            return None, empty_pts

        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        allowed = set(self._club_class_indices)

        candidate_indices = [i for i in range(len(cls_ids)) if cls_ids[i] in allowed]
        if not candidate_indices:
            return None, empty_pts

        # 許可クラス内で信頼度最大
        best_local = max(candidate_indices, key=lambda i: confs[i])
        best_i = best_local

        poly_list = res.masks.xy
        if best_i >= len(poly_list):
            return None, empty_pts

        poly = np.asarray(poly_list[best_i], dtype=np.float32)
        mask = _polygon_to_mask(poly, h, w)
        if mask.sum() < 1:
            return None, empty_pts

        pts = sample_points_from_mask(mask, n=num_samples)
        return mask, pts
