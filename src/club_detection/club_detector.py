"""
クラブ検出クラス
YOLOv8を使用してゴルフクラブを検出・トラッキング
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class ClubDetector:
    """YOLOv8を使用したゴルフクラブ検出クラス"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初期化
        
        Args:
            model_path: カスタムモデルのパス（Noneの場合はデフォルトモデル）
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # デフォルトではYOLOv8nを使用
            # カスタムモデルは後で学習する必要がある
            self.model = YOLO('yolov8n.pt')
            # TODO: ゴルフクラブ検出用のカスタムモデルを学習・配置する
    
    def detect_club(self, frame: np.ndarray) -> List[Dict]:
        """
        1フレームからクラブを検出
        
        Args:
            frame: 入力フレーム
            
        Returns:
            検出されたクラブ情報のリスト
        """
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
    
    def detect_club_in_video(self, video_path: str | Path) -> List[Dict]:
        """
        動画全体からクラブを検出・トラッキング
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            各フレームの検出結果のリスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
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
    
    def extract_club_trajectory(self, detections_list: List[Dict]) -> Dict:
        """
        検出結果からクラブの軌道を抽出
        
        Args:
            detections_list: 検出結果のリスト
            
        Returns:
            クラブの軌道情報（ヘッドとグリップの軌道）
        """
        # TODO: トラッキングIDに基づいて軌道を抽出
        # 現在は簡易実装
        trajectory = {
            'club_head': [],
            'grip': []
        }
        
        for frame_data in detections_list:
            detections = frame_data.get('detections', [])
            if detections:
                # 最初の検出を使用（複数検出の場合は改善が必要）
                detection = detections[0]
                bbox = detection['bbox']
                center = detection['center']
                
                # 簡易的にヘッド位置を推定（実際はより高度な処理が必要）
                trajectory['club_head'].append({
                    'frame': frame_data['frame_number'],
                    'position': center,
                    'bbox': bbox
                })
        
        return trajectory

