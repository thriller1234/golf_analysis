"""
姿勢推定クラス
MediaPipe Pose Landmarker（heavy）を使用して動画から骨格情報を抽出。
動画では detect_for_video を用いてトラッキング精度を上げる。
GPU が利用可能な場合は TFLite GPU デリゲートを試みる。
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from pathlib import Path

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )
    import mediapipe as mp

    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    raise ImportError(
        "MediaPipe 0.10以降が必要です。\n"
        "pip install --upgrade mediapipe を実行してください。\n"
    )


# Google のホストモデル（heavy = 最高精度、計算コスト大）
_POSE_MODEL_HEAVY_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
_DEFAULT_MODEL_REL = Path("data/models/pose_landmarker_heavy.task")


class PoseEstimator:
    """MediaPipe Pose Landmarker を用いた姿勢推定"""

    @staticmethod
    def _download_model(model_path: Path, url: str) -> None:
        import urllib.request

        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"モデルファイルをダウンロード中: {url}")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"モデルファイルをダウンロードしました: {model_path}")

    def __init__(
        self,
        *,
        use_gpu: bool = True,
        model_path: Optional[Path] = None,
        offline_only: bool = True,
    ):
        """
        Args:
            use_gpu: True のとき GPU デリゲートを試み、失敗時は CPU にフォールバック
            model_path: .task ファイル（未指定時は heavy を data/models に配置）
            offline_only: True のときファイルが無ければダウンロードしない（FileNotFoundError）。
                False のときのみ Google から heavy を取得する。
        """
        if not USE_NEW_API:
            raise RuntimeError("MediaPipe Tasks API が必要です")

        root = Path(__file__).resolve().parent.parent.parent
        self._model_path = model_path or (root / _DEFAULT_MODEL_REL)
        if not self._model_path.exists():
            if offline_only:
                raise FileNotFoundError(
                    f"Pose Landmarker の .task がありません（オフラインモード）: {self._model_path}\n"
                    "  data/models に pose_landmarker_heavy.task を置くか、"
                    " --off-line-only False で自動ダウンロードを許可してください。"
                )
            self._download_model(self._model_path, _POSE_MODEL_HEAVY_URL)

        path_str = str(self._model_path)
        delegate_attempts: List[Optional[python.BaseOptions.Delegate]] = []
        if use_gpu:
            delegate_attempts.append(python.BaseOptions.Delegate.GPU)
        delegate_attempts.append(None)

        last_error: Optional[BaseException] = None
        self.detector = None
        for delegate in delegate_attempts:
            try:
                base_opts = python.BaseOptions(model_asset_path=path_str, delegate=delegate)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=VisionTaskRunningMode.VIDEO,
                    output_segmentation_masks=False,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    num_poses=1,
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                if delegate == python.BaseOptions.Delegate.GPU:
                    print("MediaPipe Pose: GPU デリゲートで実行します。")
                else:
                    print("MediaPipe Pose: CPU で実行します（GPU が使えない環境向け）。")
                break
            except BaseException as e:
                last_error = e
                if delegate == python.BaseOptions.Delegate.GPU:
                    print(f"注意: GPU で Pose Landmarker を初期化できませんでした。CPU を試します。 ({e})")

        if self.detector is None:
            raise RuntimeError(f"Pose Landmarker を初期化できませんでした: {last_error}") from last_error

        self.use_new_api = True
        self._mp = mp

    def _result_to_dict(self, detection_result: Any) -> Optional[Dict]:
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None

        pose_landmarks = detection_result.pose_landmarks[0]
        landmarks = {}
        for idx, landmark in enumerate(pose_landmarks):
            landmarks[idx] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }

        wl = getattr(detection_result, "pose_world_landmarks", None)
        world = wl[0] if wl else None

        return {
            "landmarks": landmarks,
            "world_landmarks": world,
            "results": detection_result,
        }

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """単発画像・テスト向け（VIDEO モードでは timestamp 0 の 1 コマとして処理）。"""
        return self.process_frame_video(frame, timestamp_ms=0)

    def process_frame_video(self, frame: np.ndarray, *, timestamp_ms: int) -> Optional[Dict]:
        """動画ストリーム用。timestamp_ms は単調増加であること。"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return self._result_to_dict(detection_result)

    def process_video_dense(self, video_path: str | Path) -> List[Optional[Dict]]:
        """全フレームを走査し、検出失敗フレームは None を含むリストを返す。"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        keypoints_list: List[Optional[Dict]] = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_ms = int(round(frame_count * 1000.0 / fps))
                keypoints = self.process_frame_video(frame, timestamp_ms=timestamp_ms)
                if keypoints is not None:
                    keypoints["frame_number"] = frame_count
                keypoints_list.append(keypoints)
                frame_count += 1
        finally:
            cap.release()

        return keypoints_list

    def process_video(self, video_path: str | Path) -> List[Dict]:
        """後方互換: 検出に成功したフレームのみのリスト（frame_number は元動画のインデックス）。"""
        dense = self.process_video_dense(video_path)
        out: List[Dict] = []
        for i, k in enumerate(dense):
            if k is None:
                continue
            entry = dict(k)
            entry["frame_number"] = i
            out.append(entry)
        return out

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """レガシー互換（内部利用は非推奨）"""
        annotated_frame = frame.copy()
        if hasattr(results, "pose_landmarks") and results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            h, w = frame.shape[:2]
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
        return annotated_frame

    def get_key_joints(self, landmarks: Dict) -> Dict[str, tuple]:
        joint_mapping = {
            "nose": 0,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }

        key_joints = {}
        for name, idx in joint_mapping.items():
            if idx in landmarks:
                joint = landmarks[idx]
                key_joints[name] = (joint["x"], joint["y"])

        return key_joints

    def close(self) -> None:
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
