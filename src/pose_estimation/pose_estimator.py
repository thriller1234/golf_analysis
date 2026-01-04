"""
姿勢推定クラス
MediaPipe Poseを使用して動画から骨格情報を抽出
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# MediaPipe 0.10以降の新しいAPIを使用
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    raise ImportError(
        "MediaPipe 0.10以降が必要です。\n"
        "pip install --upgrade mediapipe を実行してください。\n"
        "または、古いAPIを使用する場合は: pip install mediapipe==0.9.3.0"
    )


class PoseEstimator:
    """MediaPipeを使用した姿勢推定クラス"""
    
    @staticmethod
    def _download_model(model_path: Path):
        """モデルファイルをダウンロード"""
        import urllib.request
        import os
        
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        
        # ディレクトリを作成
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"モデルファイルをダウンロード中: {model_url}")
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"モデルファイルをダウンロードしました: {model_path}")
    
    def __init__(self, model_complexity: int = 2):
        """
        初期化
        
        Args:
            model_complexity: MediaPipeのモデル複雑度 (0, 1, 2)
                             2が最も高精度だが計算負荷が高い
        """
        if USE_NEW_API:
            # 新しいAPI (MediaPipe 0.10以降)
            # モデルファイルのパス
            model_path = Path(__file__).parent.parent.parent / "data" / "models" / "pose_landmarker_lite.task"
            
            # モデルファイルが存在しない場合はダウンロード
            if not model_path.exists():
                self._download_model(model_path)
            
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_poses=1
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.use_new_api = True
        else:
            # 古いAPI (MediaPipe 0.9以前)
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.use_new_api = False
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        1フレームを処理して骨格情報を取得
        
        Args:
            frame: 入力フレーム (BGR形式)
            
        Returns:
            骨格情報の辞書、検出失敗時はNone
        """
        if self.use_new_api:
            # 新しいAPIを使用
            # RGBに変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipeの画像形式に変換
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 姿勢推定
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None
            
            # 最初の検出結果を使用
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # 骨格情報を辞書形式で返す
            landmarks = {}
            for idx, landmark in enumerate(pose_landmarks):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            return {
                'landmarks': landmarks,
                'world_landmarks': detection_result.pose_world_landmarks[0] if detection_result.pose_world_landmarks else None,
                'results': detection_result
            }
        else:
            # 古いAPIを使用
            # RGBに変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿勢推定
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # 骨格情報を辞書形式で返す
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            return {
                'landmarks': landmarks,
                'world_landmarks': results.pose_world_landmarks,
                'results': results
            }
    
    def process_video(self, video_path: str | Path) -> List[Dict]:
        """
        動画全体を処理して骨格情報を取得
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            各フレームの骨格情報のリスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")
        
        keypoints_list = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                keypoints = self.process_frame(frame)
                if keypoints:
                    keypoints['frame_number'] = frame_count
                    keypoints_list.append(keypoints)
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return keypoints_list
    
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        骨格情報をフレームに描画
        
        Args:
            frame: 入力フレーム
            results: MediaPipeの結果
            
        Returns:
            描画済みフレーム
        """
        annotated_frame = frame.copy()
        
        if self.use_new_api:
            # 新しいAPIの場合、手動で描画する必要がある
            # 簡易的な実装（必要に応じて改善）
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_landmarks = results.pose_landmarks[0]
                h, w = frame.shape[:2]
                
                # 主要な関節を描画
                for landmark in pose_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
        else:
            # 古いAPIの場合
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                )
            )
        
        return annotated_frame
    
    def get_key_joints(self, landmarks: Dict) -> Dict[str, Tuple[float, float]]:
        """
        主要な関節点を取得
        
        Args:
            landmarks: 骨格情報の辞書
            
        Returns:
            主要関節点の座標辞書
        """
        # MediaPipe Poseの主要な関節インデックス
        joint_mapping = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
        key_joints = {}
        for name, idx in joint_mapping.items():
            if idx in landmarks:
                joint = landmarks[idx]
                key_joints[name] = (joint['x'], joint['y'])
        
        return key_joints
    
    def __del__(self):
        """リソースの解放"""
        if hasattr(self, 'use_new_api'):
            if not self.use_new_api and hasattr(self, 'pose'):
                self.pose.close()
            elif self.use_new_api and hasattr(self, 'detector'):
                # 新しいAPIでは明示的なcloseは不要
                pass

