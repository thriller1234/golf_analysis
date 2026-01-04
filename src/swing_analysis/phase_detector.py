"""
スイング位相検出クラス
アドレス、トップ、インパクト、フィニッシュを自動検出
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


class PhaseDetector:
    """スイング位相を検出するクラス"""
    
    # MediaPipe Poseの関節インデックス
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0
    
    def __init__(self, min_stable_frames: int = 5, velocity_threshold: float = 0.01):
        """
        初期化
        
        Args:
            min_stable_frames: 静止状態と判定する最小フレーム数
            velocity_threshold: 静止状態と判定する速度の閾値
        """
        self.min_stable_frames = min_stable_frames
        self.velocity_threshold = velocity_threshold
    
    def _get_wrist_position(self, landmarks: Dict) -> Optional[Tuple[float, float]]:
        """手首の位置を取得（両手首の平均）"""
        left_wrist = landmarks.get(self.LEFT_WRIST)
        right_wrist = landmarks.get(self.RIGHT_WRIST)
        
        if left_wrist and right_wrist:
            x = (left_wrist['x'] + right_wrist['x']) / 2
            y = (left_wrist['y'] + right_wrist['y']) / 2
            return (x, y)
        elif left_wrist:
            return (left_wrist['x'], left_wrist['y'])
        elif right_wrist:
            return (right_wrist['x'], right_wrist['y'])
        return None
    
    def _calculate_wrist_heights(self, keypoints: List[Dict]) -> np.ndarray:
        """全フレームの手首の高さ（y座標）を計算"""
        heights = []
        for kp in keypoints:
            landmarks = kp.get('landmarks', {})
            wrist_pos = self._get_wrist_position(landmarks)
            if wrist_pos:
                heights.append(wrist_pos[1])  # y座標
            else:
                heights.append(np.nan)
        return np.array(heights)
    
    def _calculate_wrist_velocities(self, keypoints: List[Dict]) -> np.ndarray:
        """手首の速度を計算"""
        velocities = []
        for i in range(len(keypoints)):
            if i == 0:
                velocities.append(0.0)
                continue
            
            prev_kp = keypoints[i-1]
            curr_kp = keypoints[i]
            
            prev_pos = self._get_wrist_position(prev_kp.get('landmarks', {}))
            curr_pos = self._get_wrist_position(curr_kp.get('landmarks', {}))
            
            if prev_pos and curr_pos:
                # ユークリッド距離を速度として使用
                velocity = np.sqrt(
                    (curr_pos[0] - prev_pos[0])**2 + 
                    (curr_pos[1] - prev_pos[1])**2
                )
                velocities.append(velocity)
            else:
                velocities.append(0.0)
        
        return np.array(velocities)
    
    def _calculate_hip_rotation(self, landmarks: Dict) -> Optional[float]:
        """腰の回転角を計算（肩と腰の角度）"""
        left_shoulder = landmarks.get(self.LEFT_SHOULDER)
        right_shoulder = landmarks.get(self.RIGHT_SHOULDER)
        left_hip = landmarks.get(self.LEFT_HIP)
        right_hip = landmarks.get(self.RIGHT_HIP)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        # 肩の中点と腰の中点を計算
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_center_x = (left_hip['x'] + right_hip['x']) / 2
        hip_center_y = (left_hip['y'] + right_hip['y']) / 2
        
        # 角度を計算（水平からの角度）
        angle = np.arctan2(
            shoulder_center_y - hip_center_y,
            shoulder_center_x - hip_center_x
        )
        
        return np.degrees(angle)
    
    def detect_address(self, keypoints: List[Dict]) -> int:
        """
        アドレス位相を検出
        スイング開始前の静止状態を検出
        """
        if not keypoints:
            return 0
        
        velocities = self._calculate_wrist_velocities(keypoints)
        
        # 最初の静止状態を検出
        for i in range(len(velocities) - self.min_stable_frames):
            window = velocities[i:i+self.min_stable_frames]
            if np.all(window < self.velocity_threshold):
                return i
        
        return 0
    
    def detect_top(self, keypoints: List[Dict], address_frame: int = 0) -> int:
        """
        トップ位相を検出
        バックスイングの最高点（手首が最も高い位置）
        """
        if not keypoints:
            return 0
        
        heights = self._calculate_wrist_heights(keypoints)
        
        # アドレス以降の範囲で最小のy座標（最も高い位置）を検出
        search_start = address_frame
        # トップは通常、動画の前半部分にあるため、全体の60%までを検索範囲とする
        search_end = min(len(keypoints), int(len(keypoints) * 0.6))
        
        if search_end <= search_start:
            search_end = len(keypoints)
        
        # 検索範囲内で最小のy座標を探す
        search_heights = heights[search_start:search_end]
        if len(search_heights) == 0 or np.all(np.isnan(search_heights)):
            return search_start + len(search_heights) // 2
        
        valid_indices = ~np.isnan(search_heights)
        if not np.any(valid_indices):
            return search_start + len(search_heights) // 2
        
        min_height_idx = np.nanargmin(search_heights)
        return search_start + min_height_idx
    
    def detect_impact(self, keypoints: List[Dict], address_frame: int = 0, top_frame: Optional[int] = None) -> int:
        """
        インパクト位相を検出
        ダウンスイングで手首の速度が最大になる点を優先的に検出
        （手首の高さだけでは正確に検出できないため、速度を優先）
        
        Args:
            keypoints: 骨格情報のリスト
            address_frame: アドレス位相のフレーム番号
            top_frame: トップ位相のフレーム番号（指定されていない場合は自動検出）
        """
        if not keypoints:
            return 0
        
        heights = self._calculate_wrist_heights(keypoints)
        velocities = self._calculate_wrist_velocities(keypoints)
        
        # アドレス以降の範囲で検索
        search_start = address_frame
        search_end = len(keypoints)
        
        if search_end <= search_start:
            search_end = len(keypoints)
        
        # トップ以降のダウンスイング領域を特定
        if top_frame is None:
            top_frame = self.detect_top(keypoints, address_frame)
        downswing_start = max(search_start, top_frame)
        
        # ダウンスイング領域で速度が最大になる点を探す（より正確）
        if downswing_start < search_end:
            search_velocities = velocities[downswing_start:search_end]
            if len(search_velocities) > 0:
                # 速度が最大の点を探す
                max_velocity_idx = np.argmax(search_velocities)
                impact_candidate = downswing_start + max_velocity_idx
                
                # 速度が最大の点の前後で、手首が低い位置にあることを確認
                # （インパクト付近は手首が低い位置にあるはず）
                check_range = 10  # 前後10フレームをチェック
                check_start = max(downswing_start, impact_candidate - check_range)
                check_end = min(search_end, impact_candidate + check_range)
                
                if check_end > check_start:
                    check_heights = heights[check_start:check_end]
                    if len(check_heights) > 0 and not np.all(np.isnan(check_heights)):
                        # 速度最大点付近で最も低い位置を探す
                        local_max_height_idx = np.nanargmax(check_heights)
                        return check_start + local_max_height_idx
                
                return impact_candidate
        
        # フォールバック: 手首が最も低い位置を探す
        search_heights = heights[search_start:search_end]
        if len(search_heights) == 0 or np.all(np.isnan(search_heights)):
            return search_start + len(search_heights) // 2
        
        valid_indices = ~np.isnan(search_heights)
        if not np.any(valid_indices):
            return search_start + len(search_heights) // 2
        
        max_height_idx = np.nanargmax(search_heights)
        return search_start + max_height_idx
    
    def detect_finish(self, keypoints: List[Dict], impact_frame: int = 0) -> int:
        """
        フィニッシュ位相を検出
        インパクト後の静止状態
        """
        if not keypoints:
            return len(keypoints) - 1
        
        velocities = self._calculate_wrist_velocities(keypoints)
        
        # インパクト以降の範囲で静止状態を検出
        search_start = impact_frame
        search_end = len(keypoints)
        
        # 最後から逆順に検索
        for i in range(search_end - self.min_stable_frames, search_start, -1):
            if i < 0:
                break
            window = velocities[i:i+self.min_stable_frames]
            if len(window) == self.min_stable_frames and np.all(window < self.velocity_threshold):
                return i
        
        return len(keypoints) - 1
    
    def detect_phases(self, keypoints: List[Dict]) -> Dict[str, int]:
        """
        スイング位相を検出
        
        Args:
            keypoints: 骨格情報のリスト
            
        Returns:
            各位相のフレーム番号の辞書
        """
        if not keypoints:
            return {
                'address': 0,
                'top': 0,
                'impact': 0,
                'finish': 0
            }
        
        # 順番に検出（前の位相の情報を使用）
        address = self.detect_address(keypoints)
        top = self.detect_top(keypoints, address)
        impact = self.detect_impact(keypoints, address, top_frame=top)
        finish = self.detect_finish(keypoints, impact)
        
        # 順序の整合性を確認
        if not (address <= top <= impact <= finish):
            # 順序がおかしい場合は、アドレスを基準に再計算
            top = max(address, min(top, impact))
            impact = max(top, min(impact, finish))
            finish = max(impact, finish)
        
        return {
            'address': address,
            'top': top,
            'impact': impact,
            'finish': finish
        }

