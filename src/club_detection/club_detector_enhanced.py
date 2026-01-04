"""
クラブ検出の拡張クラス
エッジ検出と手首位置を組み合わせた簡易クラブ検出
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class EnhancedClubDetector:
    """エッジ検出と手首位置を組み合わせたクラブ検出クラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def detect_club_edges(
        self,
        frame: np.ndarray,
        wrist_position: Optional[Tuple[float, float]] = None,
        roi_size: int = 100
    ) -> List[Dict]:
        """
        エッジ検出を使用してクラブを検出
        
        Args:
            frame: 入力フレーム
            wrist_position: 手首の位置（検索範囲を絞るため）
            roi_size: 手首周辺の検索範囲（ピクセル）
            
        Returns:
            検出されたクラブエッジの情報
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 手首位置周辺のROIを設定
        if wrist_position:
            h, w = frame.shape[:2]
            wrist_x = int(wrist_position[0] * w)
            wrist_y = int(wrist_position[1] * h)
            
            x1 = max(0, wrist_x - roi_size // 2)
            y1 = max(0, wrist_y - roi_size // 2)
            x2 = min(w, wrist_x + roi_size // 2)
            y2 = min(h, wrist_y + roi_size // 2)
            
            roi_edges = edges[y1:y2, x1:x2]
        else:
            roi_edges = edges
            x1, y1 = 0, 0
        
        # 線分検出（HoughLinesP）
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        detections = []
        if lines is not None:
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                # ROI座標を元の画像座標に変換
                x1_abs = x1_line + x1
                y1_abs = y1_line + y1
                x2_abs = x2_line + x1
                y2_abs = y2_line + y1
                
                # 線分の長さと角度を計算
                length = np.sqrt((x2_abs - x1_abs)**2 + (y2_abs - y1_abs)**2)
                angle = np.arctan2(y2_abs - y1_abs, x2_abs - x1_abs) * 180 / np.pi
                
                # クラブらしい線分をフィルタリング（長さと角度で判定）
                if 20 < length < 200 and -80 < angle < 80:
                    detections.append({
                        'line': [x1_abs, y1_abs, x2_abs, y2_abs],
                        'length': length,
                        'angle': angle,
                        'center': [(x1_abs + x2_abs) / 2, (y1_abs + y2_abs) / 2]
                    })
        
        return detections
    
    def estimate_club_from_wrist_and_edges(
        self,
        frame: np.ndarray,
        wrist_position: Tuple[float, float],
        keypoints: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        手首位置とエッジ検出を組み合わせてクラブを推定
        
        Args:
            frame: 入力フレーム
            wrist_position: 手首の位置
            keypoints: 骨格情報（オプション）
            
        Returns:
            クラブの推定情報
        """
        # エッジ検出
        edges = self.detect_club_edges(frame, wrist_position)
        
        if not edges:
            return None
        
        # 手首に最も近い線分をクラブとして選択
        h, w = frame.shape[:2]
        wrist_x = int(wrist_position[0] * w)
        wrist_y = int(wrist_position[1] * h)
        
        min_distance = float('inf')
        best_edge = None
        
        for edge in edges:
            center = edge['center']
            distance = np.sqrt((center[0] - wrist_x)**2 + (center[1] - wrist_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_edge = edge
        
        if best_edge:
            line = best_edge['line']
            # 線分の端をクラブヘッドとグリップとして推定
            # 手首に近い方をグリップ、遠い方をヘッドとする
            dist1 = np.sqrt((line[0] - wrist_x)**2 + (line[1] - wrist_y)**2)
            dist2 = np.sqrt((line[2] - wrist_x)**2 + (line[3] - wrist_y)**2)
            
            if dist1 < dist2:
                grip_pos = [line[0] / w, line[1] / h]
                head_pos = [line[2] / w, line[3] / h]
            else:
                grip_pos = [line[2] / w, line[3] / h]
                head_pos = [line[0] / w, line[1] / h]
            
            return {
                'grip_position': grip_pos,
                'head_position': head_pos,
                'length': best_edge['length'],
                'angle': best_edge['angle'],
                'confidence': 0.6  # 簡易検出なので信頼度は中程度
            }
        
        return None

