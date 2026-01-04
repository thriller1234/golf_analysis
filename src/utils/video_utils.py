"""
動画処理ユーティリティ
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class VideoUtils:
    """動画処理のユーティリティクラス"""
    
    @staticmethod
    def get_video_info(video_path: str | Path) -> dict:
        """
        動画の情報を取得
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            動画情報の辞書
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
        finally:
            cap.release()
        
        return info
    
    @staticmethod
    def extract_frame(video_path: str | Path, frame_number: int) -> Optional[np.ndarray]:
        """
        指定フレームを抽出
        
        Args:
            video_path: 動画ファイルのパス
            frame_number: フレーム番号
            
        Returns:
            フレーム画像、失敗時はNone
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            return frame if ret else None
        finally:
            cap.release()

