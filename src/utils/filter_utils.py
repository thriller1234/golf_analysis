"""
フィルタリングユーティリティ
Kalman Filterや移動平均など
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple


class FilterUtils:
    """フィルタリングのユーティリティクラス"""
    
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        移動平均フィルタ
        
        Args:
            data: 入力データ
            window_size: ウィンドウサイズ
            
        Returns:
            フィルタリングされたデータ
        """
        if len(data) < window_size:
            return data
        
        # パディング
        padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        
        # 移動平均
        filtered = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
        
        return filtered
    
    @staticmethod
    def kalman_filter_2d(
        measurements: List[Tuple[float, float]],
        process_noise: float = 0.1,
        measurement_noise: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        2D座標にKalman Filterを適用
        
        Args:
            measurements: 測定値のリスト [(x, y), ...]
            process_noise: プロセスノイズ
            measurement_noise: 測定ノイズ
            
        Returns:
            フィルタリングされた座標のリスト
        """
        if not measurements:
            return []
        
        # Kalman Filterの初期化（位置と速度を追跡）
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # 状態遷移行列（等速直線運動モデル）
        dt = 1.0  # タイムステップ
        kf.F = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        
        # 観測行列
        kf.H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        
        # 共分散行列
        kf.P *= 1000.
        kf.R = np.eye(2) * measurement_noise
        kf.Q = np.eye(4) * process_noise
        
        # 初期状態
        if measurements:
            kf.x = np.array([measurements[0][0], measurements[0][1], 0., 0.])
        
        filtered = []
        for measurement in measurements:
            kf.predict()
            kf.update(np.array([measurement[0], measurement[1]]))
            filtered.append((kf.x[0], kf.x[1]))
        
        return filtered

