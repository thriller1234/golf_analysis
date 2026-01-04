"""
スイング正規化クラス
空間・時間の正規化を行う
"""

from typing import List, Dict
import numpy as np


class SwingNormalizer:
    """スイングの正規化を行うクラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def normalize_spatial(
        self,
        keypoints: List[Dict],
        reference_keypoints: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        空間の正規化（身長・スケール調整）
        
        Args:
            keypoints: 正規化する骨格情報
            reference_keypoints: 基準となる骨格情報（オプション）
            
        Returns:
            正規化された骨格情報
        """
        # TODO: 実装
        # - 腰の幅や肩の幅を基準にしてスケール調整
        # - 骨格スケールの正規化
        
        return keypoints
    
    def normalize_temporal(
        self,
        keypoints: List[Dict],
        target_length: int
    ) -> List[Dict]:
        """
        時間の正規化（位相同期）
        
        Args:
            keypoints: 正規化する骨格情報
            target_length: 目標フレーム数
            
        Returns:
            正規化された骨格情報
        """
        # TODO: 実装
        # - 補間処理でフレーム数を調整
        # - 位相で揃える
        
        if len(keypoints) == target_length:
            return keypoints
        
        # 簡易的な線形補間
        normalized = []
        for i in range(target_length):
            idx = i * len(keypoints) / target_length
            idx = int(idx)
            if idx >= len(keypoints):
                idx = len(keypoints) - 1
            normalized.append(keypoints[idx])
        
        return normalized

