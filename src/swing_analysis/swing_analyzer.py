"""
スイング解析クラス
プロ選手との比較解析を行う
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from .phase_detector import PhaseDetector
from .normalizer import SwingNormalizer


class SwingAnalyzer:
    """スイング解析と比較を行うクラス"""
    
    # MediaPipe Poseの関節インデックス
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    def __init__(self):
        """初期化"""
        self.phase_detector = PhaseDetector()
        self.normalizer = SwingNormalizer()
    
    def _get_joint_position(self, landmarks: Dict, joint_idx: int) -> Optional[Tuple[float, float]]:
        """関節の位置を取得"""
        joint = landmarks.get(joint_idx)
        if joint:
            return (joint['x'], joint['y'])
        return None
    
    def _calculate_rotation_angle(
        self, 
        landmarks: Dict, 
        start_joint: int, 
        end_joint: int
    ) -> Optional[float]:
        """2つの関節間の角度を計算"""
        start_pos = self._get_joint_position(landmarks, start_joint)
        end_pos = self._get_joint_position(landmarks, end_joint)
        
        if not start_pos or not end_pos:
            return None
        
        angle = np.arctan2(
            end_pos[1] - start_pos[1],
            end_pos[0] - start_pos[0]
        )
        return np.degrees(angle)
    
    def extract_features(self, keypoints: List[Dict], phases: Optional[Dict[str, int]] = None) -> Dict:
        """
        スイングから特徴量を抽出
        
        Args:
            keypoints: 骨格情報のリスト
            phases: スイング位相の辞書（オプション）
            
        Returns:
            特徴量の辞書
        """
        if not keypoints:
            return {}
        
        # 位相を検出（指定されていない場合）
        if phases is None:
            phases = self.phase_detector.detect_phases(keypoints)
        
        features = {
            'phases': phases,
            'shoulder_rotation': [],
            'hip_rotation': [],
            'x_factor': [],
            'wrist_trajectory': [],
            'wrist_heights': [],
        }
        
        # 各フレームの特徴量を計算
        for kp in keypoints:
            landmarks = kp.get('landmarks', {})
            
            # 肩の回転角
            shoulder_angle = self._calculate_rotation_angle(
                landmarks, self.LEFT_SHOULDER, self.RIGHT_SHOULDER
            )
            features['shoulder_rotation'].append(shoulder_angle)
            
            # 腰の回転角
            hip_angle = self._calculate_rotation_angle(
                landmarks, self.LEFT_HIP, self.RIGHT_HIP
            )
            features['hip_rotation'].append(hip_angle)
            
            # Xファクター（肩と腰の回転角の差）
            if shoulder_angle is not None and hip_angle is not None:
                x_factor = abs(shoulder_angle - hip_angle)
                features['x_factor'].append(x_factor)
            else:
                features['x_factor'].append(None)
            
            # 手首の位置
            left_wrist = self._get_joint_position(landmarks, self.LEFT_WRIST)
            right_wrist = self._get_joint_position(landmarks, self.RIGHT_WRIST)
            
            if left_wrist and right_wrist:
                wrist_center = (
                    (left_wrist[0] + right_wrist[0]) / 2,
                    (left_wrist[1] + right_wrist[1]) / 2
                )
                features['wrist_trajectory'].append(wrist_center)
                features['wrist_heights'].append(wrist_center[1])
            else:
                features['wrist_trajectory'].append(None)
                features['wrist_heights'].append(None)
        
        # 位相ごとの特徴量を計算
        phase_features = {}
        for phase_name, frame_num in phases.items():
            if 0 <= frame_num < len(keypoints):
                phase_features[phase_name] = {
                    'frame': frame_num,
                    'shoulder_rotation': features['shoulder_rotation'][frame_num],
                    'hip_rotation': features['hip_rotation'][frame_num],
                    'x_factor': features['x_factor'][frame_num],
                    'wrist_height': features['wrist_heights'][frame_num],
                }
        
        features['phase_features'] = phase_features
        
        return features
    
    def compare_swings(
        self,
        my_swing: str | Path,
        pro_swing: str | Path,
        my_keypoints: Optional[List[Dict]] = None,
        pro_keypoints: Optional[List[Dict]] = None
    ) -> Dict:
        """
        自分のスイングとプロのスイングを比較
        
        Args:
            my_swing: 自分のスイング動画のパス
            pro_swing: プロのスイング動画のパス
            my_keypoints: 自分のスイングの骨格情報（オプション）
            pro_keypoints: プロのスイングの骨格情報（オプション）
            
        Returns:
            比較結果の辞書
        """
        from src.pose_estimation import PoseEstimator
        
        # 骨格情報を取得（指定されていない場合）
        if my_keypoints is None:
            estimator = PoseEstimator()
            my_keypoints = estimator.process_video(my_swing)
        
        if pro_keypoints is None:
            estimator = PoseEstimator()
            pro_keypoints = estimator.process_video(pro_swing)
        
        # 位相を検出
        my_phases = self.phase_detector.detect_phases(my_keypoints)
        pro_phases = self.phase_detector.detect_phases(pro_keypoints)
        
        # 特徴量を抽出
        my_features = self.extract_features(my_keypoints, my_phases)
        pro_features = self.extract_features(pro_keypoints, pro_phases)
        
        # 比較結果を計算
        comparison = {
            'my_phases': my_phases,
            'pro_phases': pro_phases,
            'phase_comparison': {},
            'feature_differences': {}
        }
        
        # 位相ごとの比較
        for phase_name in ['address', 'top', 'impact', 'finish']:
            my_phase_feat = my_features['phase_features'].get(phase_name, {})
            pro_phase_feat = pro_features['phase_features'].get(phase_name, {})
            
            if my_phase_feat and pro_phase_feat:
                comparison['phase_comparison'][phase_name] = {
                    'shoulder_rotation_diff': (
                        my_phase_feat.get('shoulder_rotation') - 
                        pro_phase_feat.get('shoulder_rotation')
                        if my_phase_feat.get('shoulder_rotation') is not None and 
                           pro_phase_feat.get('shoulder_rotation') is not None
                        else None
                    ),
                    'hip_rotation_diff': (
                        my_phase_feat.get('hip_rotation') - 
                        pro_phase_feat.get('hip_rotation')
                        if my_phase_feat.get('hip_rotation') is not None and 
                           pro_phase_feat.get('hip_rotation') is not None
                        else None
                    ),
                    'x_factor_diff': (
                        my_phase_feat.get('x_factor') - 
                        pro_phase_feat.get('x_factor')
                        if my_phase_feat.get('x_factor') is not None and 
                           pro_phase_feat.get('x_factor') is not None
                        else None
                    ),
                    'wrist_height_diff': (
                        my_phase_feat.get('wrist_height') - 
                        pro_phase_feat.get('wrist_height')
                        if my_phase_feat.get('wrist_height') is not None and 
                           pro_phase_feat.get('wrist_height') is not None
                        else None
                    ),
                }
        
        # 全体的な特徴量の差分
        my_x_factors = [x for x in my_features['x_factor'] if x is not None]
        pro_x_factors = [x for x in pro_features['x_factor'] if x is not None]
        
        comparison['feature_differences'] = {
            'max_x_factor_my': max(my_x_factors) if my_x_factors else None,
            'max_x_factor_pro': max(pro_x_factors) if pro_x_factors else None,
            'max_x_factor_diff': (
                max(my_x_factors) - max(pro_x_factors)
                if my_x_factors and pro_x_factors else None
            ),
        }
        
        return {
            'my_swing': str(my_swing),
            'pro_swing': str(pro_swing),
            'my_features': my_features,
            'pro_features': pro_features,
            'comparison': comparison
        }

