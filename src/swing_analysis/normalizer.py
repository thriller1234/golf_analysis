"""
スイング正規化クラス
空間・時間の正規化を行う
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d


class SwingNormalizer:
    """スイングの正規化を行うクラス"""
    
    # MediaPipe Poseの関節インデックス
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(self):
        """初期化"""
        pass
    
    def _calculate_shoulder_width(self, landmarks: Dict) -> Optional[float]:
        """肩幅を計算"""
        left_shoulder = landmarks.get(self.LEFT_SHOULDER)
        right_shoulder = landmarks.get(self.RIGHT_SHOULDER)
        
        if left_shoulder and right_shoulder:
            dx = right_shoulder['x'] - left_shoulder['x']
            dy = right_shoulder['y'] - left_shoulder['y']
            return np.sqrt(dx**2 + dy**2)
        return None
    
    def _calculate_hip_width(self, landmarks: Dict) -> Optional[float]:
        """腰幅を計算"""
        left_hip = landmarks.get(self.LEFT_HIP)
        right_hip = landmarks.get(self.RIGHT_HIP)
        
        if left_hip and right_hip:
            dx = right_hip['x'] - left_hip['x']
            dy = right_hip['y'] - left_hip['y']
            return np.sqrt(dx**2 + dy**2)
        return None
    
    def _calculate_body_height(self, landmarks: Dict) -> Optional[float]:
        """身長を推定（頭から足首までの距離）"""
        # 頭の位置（鼻を使用）
        nose = landmarks.get(0)  # 鼻
        left_ankle = landmarks.get(self.LEFT_ANKLE)
        right_ankle = landmarks.get(self.RIGHT_ANKLE)
        
        if nose and (left_ankle or right_ankle):
            # 両足首の平均位置を使用
            if left_ankle and right_ankle:
                ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
            elif left_ankle:
                ankle_y = left_ankle['y']
            else:
                ankle_y = right_ankle['y']
            
            return abs(nose['y'] - ankle_y)
        return None
    
    def _get_reference_scale(
        self, 
        keypoints: List[Dict],
        use_shoulder: bool = True
    ) -> Optional[float]:
        """
        基準スケールを計算
        
        Args:
            keypoints: 骨格情報のリスト
            use_shoulder: Trueの場合は肩幅、Falseの場合は腰幅を使用
            
        Returns:
            基準スケール値
        """
        scales = []
        
        for kp in keypoints:
            landmarks = kp.get('landmarks', {})
            
            if use_shoulder:
                scale = self._calculate_shoulder_width(landmarks)
            else:
                scale = self._calculate_hip_width(landmarks)
            
            if scale is not None:
                scales.append(scale)
        
        if scales:
            # 中央値を基準として使用（外れ値の影響を減らす）
            return np.median(scales)
        
        return None
    
    def normalize_spatial(
        self,
        keypoints: List[Dict],
        reference_keypoints: Optional[List[Dict]] = None,
        target_scale: Optional[float] = None,
        use_shoulder_width: bool = True
    ) -> List[Dict]:
        """
        空間の正規化（身長・スケール調整）
        
        Args:
            keypoints: 正規化する骨格情報
            reference_keypoints: 基準となる骨格情報（オプション）
            target_scale: 目標スケール値（指定されていない場合はreference_keypointsから計算）
            use_shoulder_width: Trueの場合は肩幅、Falseの場合は腰幅を使用
            
        Returns:
            正規化された骨格情報
        """
        if not keypoints:
            return keypoints
        
        # 基準スケールを決定
        if target_scale is None:
            if reference_keypoints:
                target_scale = self._get_reference_scale(reference_keypoints, use_shoulder_width)
            else:
                # 参照がない場合は、自分自身の平均スケールを基準とする
                target_scale = self._get_reference_scale(keypoints, use_shoulder_width)
        
        if target_scale is None or target_scale == 0:
            # スケールが計算できない場合はそのまま返す
            return keypoints
        
        # 各フレームを正規化
        normalized_keypoints = []
        
        for kp in keypoints:
            landmarks = kp.get('landmarks', {})
            
            # 現在のスケールを計算
            if use_shoulder_width:
                current_scale = self._calculate_shoulder_width(landmarks)
            else:
                current_scale = self._calculate_hip_width(landmarks)
            
            if current_scale is None or current_scale == 0:
                # スケールが計算できない場合はそのまま追加
                normalized_keypoints.append(kp)
                continue
            
            # スケール比を計算
            scale_ratio = target_scale / current_scale
            
            # 腰の中心を基準点として使用
            left_hip = landmarks.get(self.LEFT_HIP)
            right_hip = landmarks.get(self.RIGHT_HIP)
            
            if left_hip and right_hip:
                center_x = (left_hip['x'] + right_hip['x']) / 2
                center_y = (left_hip['y'] + right_hip['y']) / 2
            else:
                # 腰が検出できない場合は、すべての関節の中心を使用
                valid_joints = [j for j in landmarks.values() if j]
                if valid_joints:
                    center_x = np.mean([j['x'] for j in valid_joints])
                    center_y = np.mean([j['y'] for j in valid_joints])
                else:
                    normalized_keypoints.append(kp)
                    continue
            
            # すべての関節をスケール調整
            normalized_landmarks = {}
            for idx, landmark in landmarks.items():
                # 基準点からの相対位置を計算
                rel_x = landmark['x'] - center_x
                rel_y = landmark['y'] - center_y
                
                # スケール調整
                normalized_rel_x = rel_x * scale_ratio
                normalized_rel_y = rel_y * scale_ratio
                
                # 基準点を中心に戻す
                normalized_landmarks[idx] = {
                    'x': center_x + normalized_rel_x,
                    'y': center_y + normalized_rel_y,
                    'z': landmark.get('z', 0.0),  # z座標はそのまま
                    'visibility': landmark.get('visibility', 1.0)
                }
            
            # 正規化されたキーポイントを作成
            normalized_kp = kp.copy()
            normalized_kp['landmarks'] = normalized_landmarks
            normalized_keypoints.append(normalized_kp)
        
        return normalized_keypoints
    
    def normalize_temporal(
        self,
        keypoints: List[Dict],
        target_length: int,
        phases: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """
        時間の正規化（位相同期）
        
        Args:
            keypoints: 正規化する骨格情報
            target_length: 目標フレーム数
            phases: スイング位相の辞書（位相ベースの正規化に使用）
            
        Returns:
            正規化された骨格情報
        """
        if len(keypoints) == target_length:
            return keypoints
        
        if phases and len(keypoints) > 0:
            # 位相ベースの正規化
            return self._normalize_by_phases(keypoints, target_length, phases)
        else:
            # 線形補間による正規化
            return self._normalize_linear(keypoints, target_length)
    
    def _normalize_linear(self, keypoints: List[Dict], target_length: int) -> List[Dict]:
        """線形補間による時間正規化"""
        if len(keypoints) == 0:
            return keypoints
        
        normalized = []
        source_indices = np.linspace(0, len(keypoints) - 1, target_length)
        
        for target_idx in source_indices:
            idx_low = int(np.floor(target_idx))
            idx_high = min(int(np.ceil(target_idx)), len(keypoints) - 1)
            
            if idx_low == idx_high:
                normalized.append(keypoints[idx_low])
            else:
                # 線形補間
                alpha = target_idx - idx_low
                interpolated = self._interpolate_keypoints(
                    keypoints[idx_low],
                    keypoints[idx_high],
                    alpha
                )
                normalized.append(interpolated)
        
        return normalized
    
    def _normalize_by_phases(
        self,
        keypoints: List[Dict],
        target_length: int,
        phases: Dict[str, int]
    ) -> List[Dict]:
        """位相ベースの時間正規化"""
        if len(keypoints) == 0:
            return keypoints
        
        # 各位相間のフレーム数を計算
        address = phases.get('address', 0)
        top = phases.get('top', len(keypoints) // 3)
        impact = phases.get('impact', len(keypoints) * 2 // 3)
        finish = phases.get('finish', len(keypoints) - 1)
        
        # 目標フレーム数を各位相間で分配
        target_address_to_top = int(target_length * 0.3)
        target_top_to_impact = int(target_length * 0.4)
        target_impact_to_finish = int(target_length * 0.3)
        
        normalized = []
        
        # アドレス → トップ
        if top > address:
            source_segment = keypoints[address:top+1]
            target_segment = self._normalize_linear(source_segment, target_address_to_top)
            normalized.extend(target_segment)
        
        # トップ → インパクト
        if impact > top:
            source_segment = keypoints[top:impact+1]
            target_segment = self._normalize_linear(source_segment, target_top_to_impact)
            normalized.extend(target_segment[1:])  # トップを重複させない
        
        # インパクト → フィニッシュ
        if finish > impact:
            source_segment = keypoints[impact:finish+1]
            target_segment = self._normalize_linear(source_segment, target_impact_to_finish)
            normalized.extend(target_segment[1:])  # インパクトを重複させない
        
        # 長さを調整
        if len(normalized) > target_length:
            normalized = normalized[:target_length]
        elif len(normalized) < target_length:
            # 不足分を補完
            while len(normalized) < target_length:
                normalized.append(normalized[-1] if normalized else keypoints[-1])
        
        return normalized
    
    def _interpolate_keypoints(
        self,
        kp1: Dict,
        kp2: Dict,
        alpha: float
    ) -> Dict:
        """2つのキーポイントを補間"""
        landmarks1 = kp1.get('landmarks', {})
        landmarks2 = kp2.get('landmarks', {})
        
        interpolated_landmarks = {}
        
        # 両方に存在する関節を補間
        common_indices = set(landmarks1.keys()) & set(landmarks2.keys())
        
        for idx in common_indices:
            lm1 = landmarks1[idx]
            lm2 = landmarks2[idx]
            
            interpolated_landmarks[idx] = {
                'x': lm1['x'] * (1 - alpha) + lm2['x'] * alpha,
                'y': lm1['y'] * (1 - alpha) + lm2['y'] * alpha,
                'z': lm1.get('z', 0.0) * (1 - alpha) + lm2.get('z', 0.0) * alpha,
                'visibility': max(lm1.get('visibility', 0.0), lm2.get('visibility', 0.0))
            }
        
        # kp1にのみ存在する関節
        for idx in set(landmarks1.keys()) - common_indices:
            interpolated_landmarks[idx] = landmarks1[idx].copy()
        
        # kp2にのみ存在する関節
        for idx in set(landmarks2.keys()) - common_indices:
            interpolated_landmarks[idx] = landmarks2[idx].copy()
        
        interpolated_kp = kp1.copy()
        interpolated_kp['landmarks'] = interpolated_landmarks
        
        return interpolated_kp

