"""
クラブ検出クラス
YOLOv8を使用してゴルフクラブを検出・トラッキング
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralyticsがインストールされていません。YOLOv8機能は使用できません。")


class ClubDetector:
    """YOLOv8を使用したゴルフクラブ検出クラス"""
    
    def __init__(self, model_path: Optional[str] = None, use_yolo: bool = True):
        """
        初期化
        
        Args:
            model_path: カスタムモデルのパス（Noneの場合はデフォルトモデル）
            use_yolo: YOLOv8を使用するか（Falseの場合は簡易検出を使用）
        """
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        if self.use_yolo:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # デフォルトではYOLOv8nを使用
                # カスタムモデルは後で学習する必要がある
                try:
                    self.model = YOLO('yolov8n.pt')
                except Exception as e:
                    print(f"警告: YOLOv8モデルの読み込みに失敗しました: {e}")
                    print("簡易検出モードにフォールバックします。")
                    self.use_yolo = False
        else:
            self.model = None
            print("簡易検出モードを使用します（手首位置ベース）。")
    
    def detect_club(self, frame: np.ndarray, wrist_position: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        1フレームからクラブを検出
        
        Args:
            frame: 入力フレーム
            wrist_position: 手首の位置（簡易検出モードで使用）
            
        Returns:
            検出されたクラブ情報のリスト
        """
        if self.use_yolo and self.model:
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # クラスIDと信頼度を取得
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # バウンディングボックス座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': cls_id,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
            
            return detections
        else:
            # 簡易検出モード（手首位置ベース）
            if wrist_position:
                h, w = frame.shape[:2]
                wrist_x = int(wrist_position[0] * w)
                wrist_y = int(wrist_position[1] * h)
                
                # 手首位置を中心とした簡易的なバウンディングボックス
                bbox_size = 50
                x1 = max(0, wrist_x - bbox_size)
                y1 = max(0, wrist_y - bbox_size)
                x2 = min(w, wrist_x + bbox_size)
                y2 = min(h, wrist_y + bbox_size)
                
                return [{
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.5,  # 簡易検出なので信頼度は低め
                    'class_id': -1,  # クラスIDなし
                    'center': [wrist_x, wrist_y]
                }]
            
            return []
    
    def detect_club_in_video(
        self, 
        video_path: str | Path,
        keypoints: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        動画全体からクラブを検出・トラッキング
        
        Args:
            video_path: 動画ファイルのパス
            keypoints: 骨格情報のリスト（簡易検出モードで使用）
            
        Returns:
            各フレームの検出結果のリスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        if self.use_yolo and self.model:
            # YOLOv8のトラッキング機能を使用
            results = self.model.track(
                source=str(video_path),
                persist=True,
                verbose=False
            )
            
            detections_list = []
            for frame_idx, result in enumerate(results):
                frame_detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # トラッキングIDを取得（存在する場合）
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])
                        
                        frame_detections.append({
                            'frame_number': frame_idx,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': cls_id,
                            'track_id': track_id,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
                
                detections_list.append({
                    'frame_number': frame_idx,
                    'detections': frame_detections
                })
            
            return detections_list
        else:
            # 簡易検出モード（手首位置ベース）
            if keypoints is None:
                raise ValueError("簡易検出モードではkeypointsが必要です")
            
            detections_list = []
            RIGHT_WRIST = 16
            
            for frame_idx, kp in enumerate(keypoints):
                landmarks = kp.get('landmarks', {})
                wrist = landmarks.get(RIGHT_WRIST)
                
                frame_detections = []
                if wrist:
                    wrist_pos = (wrist['x'], wrist['y'])
                    detections = self.detect_club(None, wrist_position=wrist_pos)
                    for det in detections:
                        det['frame_number'] = frame_idx
                        frame_detections.append(det)
                
                detections_list.append({
                    'frame_number': frame_idx,
                    'detections': frame_detections
                })
            
            return detections_list
    
    def extract_club_trajectory(self, detections_list: List[Dict]) -> Dict:
        """
        検出結果からクラブの軌道を抽出
        
        Args:
            detections_list: 検出結果のリスト
            
        Returns:
            クラブの軌道情報（ヘッドとグリップの軌道）
        """
        trajectory = {
            'club_head': [],
            'grip': []
        }
        
        # トラッキングIDに基づいて軌道を抽出
        track_trajectories = {}  # track_id -> list of detections
        
        for frame_data in detections_list:
            detections = frame_data.get('detections', [])
            for detection in detections:
                track_id = detection.get('track_id')
                if track_id is not None:
                    if track_id not in track_trajectories:
                        track_trajectories[track_id] = []
                    track_trajectories[track_id].append({
                        'frame': frame_data['frame_number'],
                        'detection': detection
                    })
        
        # 最も長いトラッキングをクラブとして使用
        if track_trajectories:
            longest_track_id = max(track_trajectories.keys(), 
                                 key=lambda tid: len(track_trajectories[tid]))
            track_data = track_trajectories[longest_track_id]
            
            for item in track_data:
                detection = item['detection']
                bbox = detection['bbox']
                center = detection['center']
                
                # バウンディングボックスからクラブの向きを推定
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # クラブの長い方の端をヘッド、短い方をグリップと仮定
                if width > height:
                    # 横長の場合
                    club_head_x = bbox[2] if center[0] > (bbox[0] + bbox[2]) / 2 else bbox[0]
                    club_head_y = center[1]
                    grip_x = bbox[0] if center[0] > (bbox[0] + bbox[2]) / 2 else bbox[2]
                    grip_y = center[1]
                else:
                    # 縦長の場合
                    club_head_x = center[0]
                    club_head_y = bbox[1] if center[1] > (bbox[1] + bbox[3]) / 2 else bbox[3]
                    grip_x = center[0]
                    grip_y = bbox[3] if center[1] > (bbox[1] + bbox[3]) / 2 else bbox[1]
                
                trajectory['club_head'].append({
                    'frame': item['frame'],
                    'position': [club_head_x, club_head_y],
                    'bbox': bbox,
                    'confidence': detection['confidence']
                })
                
                trajectory['grip'].append({
                    'frame': item['frame'],
                    'position': [grip_x, grip_y],
                    'bbox': bbox,
                    'confidence': detection['confidence']
                })
        else:
            # トラッキングIDがない場合のフォールバック
            for frame_data in detections_list:
                detections = frame_data.get('detections', [])
                if detections:
                    detection = detections[0]
                    bbox = detection['bbox']
                    center = detection['center']
                    
                    trajectory['club_head'].append({
                        'frame': frame_data['frame_number'],
                        'position': center,
                        'bbox': bbox,
                        'confidence': detection.get('confidence', 0.0)
                    })
        
        return trajectory
    
    def detect_club_from_wrist(
        self, 
        keypoints: List[Dict],
        wrist_joint_idx: int = 16  # 右手首
    ) -> Dict:
        """
        手首の位置からクラブを簡易的に推定（カスタムモデルがない場合のフォールバック）
        
        Args:
            keypoints: 骨格情報のリスト
            wrist_joint_idx: 手首の関節インデックス（デフォルト: 16=右手首）
            
        Returns:
            クラブの軌道情報
        """
        trajectory = {
            'club_head': [],
            'grip': []
        }
        
        for i, kp in enumerate(keypoints):
            landmarks = kp.get('landmarks', {})
            wrist = landmarks.get(wrist_joint_idx)
            
            if wrist:
                wrist_x = wrist['x']
                wrist_y = wrist['y']
                
                # 手首の位置をグリップ位置として使用
                trajectory['grip'].append({
                    'frame': i,
                    'position': [wrist_x, wrist_y],
                    'confidence': wrist.get('visibility', 0.0)
                })
                
                # クラブヘッドは手首から一定距離の位置を推定
                # 簡易的な推定（実際のクラブ角度は不明）
                estimated_head_x = wrist_x + 0.1  # 右方向にオフセット
                estimated_head_y = wrist_y + 0.05  # 下方向にオフセット
                
                trajectory['club_head'].append({
                    'frame': i,
                    'position': [estimated_head_x, estimated_head_y],
                    'confidence': wrist.get('visibility', 0.0) * 0.7  # 推定値なので信頼度を下げる
                })
        
        return trajectory

