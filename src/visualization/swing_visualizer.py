"""
スイング可視化クラス
スイング軌道や比較結果を可視化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# MediaPipeの接続情報
POSE_CONNECTIONS = [
    # 顔
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # 胴体
    (9, 10),  # 肩
    (11, 12),  # 肩
    (11, 13), (13, 15),  # 左腕
    (12, 14), (14, 16),  # 右腕
    (11, 23), (12, 24),  # 肩から腰
    (23, 24),  # 腰
    (23, 25), (25, 27),  # 左足
    (24, 26), (26, 28),  # 右足
]


class SwingVisualizer:
    """スイングの可視化を行うクラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def visualize_keypoints(
        self,
        frame: np.ndarray,
        keypoints: Dict,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        骨格情報をフレームに描画
        
        Args:
            frame: 入力フレーム
            keypoints: 骨格情報
            draw_connections: 関節間の線を描画するか
            
        Returns:
            描画済みフレーム
        """
        annotated_frame = frame.copy()
        landmarks = keypoints.get('landmarks', {})
        
        if not landmarks:
            return annotated_frame
        
        h, w = frame.shape[:2]
        
        # 関節点を描画
        joint_points = {}
        for idx, landmark in landmarks.items():
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            visibility = landmark.get('visibility', 1.0)
            
            # 可視性が低い場合は描画しない
            if visibility < 0.5:
                continue
            
            joint_points[idx] = (x, y)
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
        
        # 接続線を描画
        if draw_connections:
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx in joint_points and end_idx in joint_points:
                    pt1 = joint_points[start_idx]
                    pt2 = joint_points[end_idx]
                    cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)
        
        return annotated_frame
    
    def plot_swing_trajectory(
        self,
        keypoints: List[Dict],
        phases: Optional[Dict[str, int]] = None,
        save_path: Optional[str | Path] = None
    ):
        """
        スイング軌道をプロット
        
        Args:
            keypoints: 骨格情報のリスト
            phases: スイング位相の辞書（オプション）
            save_path: 保存パス（オプション）
        """
        if not keypoints:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Golf Swing Trajectory Analysis', fontsize=16, fontweight='bold')
        
        # 手首の軌道を抽出
        left_wrist_trajectory = []
        right_wrist_trajectory = []
        wrist_heights = []
        
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        for i, kp in enumerate(keypoints):
            landmarks = kp.get('landmarks', {})
            
            left_wrist = landmarks.get(LEFT_WRIST)
            right_wrist = landmarks.get(RIGHT_WRIST)
            
            if left_wrist:
                left_wrist_trajectory.append((left_wrist['x'], left_wrist['y']))
            if right_wrist:
                right_wrist_trajectory.append((right_wrist['x'], right_wrist['y']))
            
            # 手首の平均高さ
            if left_wrist and right_wrist:
                avg_height = (left_wrist['y'] + right_wrist['y']) / 2
                wrist_heights.append(avg_height)
            elif left_wrist:
                wrist_heights.append(left_wrist['y'])
            elif right_wrist:
                wrist_heights.append(right_wrist['y'])
            else:
                wrist_heights.append(np.nan)
        
        # 1. 手首の軌道（2D）
        ax1 = axes[0, 0]
        if left_wrist_trajectory:
            left_x, left_y = zip(*left_wrist_trajectory)
            ax1.plot(left_x, left_y, 'b-', label='Left Wrist', linewidth=2, alpha=0.7)
            ax1.scatter(left_x[0], left_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
            ax1.scatter(left_x[-1], left_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        if right_wrist_trajectory:
            right_x, right_y = zip(*right_wrist_trajectory)
            ax1.plot(right_x, right_y, 'r-', label='Right Wrist', linewidth=2, alpha=0.7)
            if not left_wrist_trajectory:
                ax1.scatter(right_x[0], right_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
                ax1.scatter(right_x[-1], right_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('X Coordinate (Normalized)')
        ax1.set_ylabel('Y Coordinate (Normalized)')
        ax1.set_title('Wrist Trajectory (2D)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Y軸を反転（画像座標系に合わせる）
        
        # 2. 手首の高さの変化
        ax2 = axes[0, 1]
        frames = np.arange(len(wrist_heights))
        ax2.plot(frames, wrist_heights, 'g-', linewidth=2, label='Wrist Height')
        
        # 位相をマーク
        if phases:
            colors = {'address': 'blue', 'top': 'red', 'impact': 'orange', 'finish': 'purple'}
            labels = {'address': 'Address', 'top': 'Top', 'impact': 'Impact', 'finish': 'Finish'}
            
            for phase_name, frame_num in phases.items():
                if 0 <= frame_num < len(wrist_heights):
                    ax2.axvline(x=frame_num, color=colors[phase_name], linestyle='--', 
                              linewidth=2, alpha=0.7, label=labels[phase_name])
                    if not np.isnan(wrist_heights[frame_num]):
                        ax2.scatter(frame_num, wrist_heights[frame_num], 
                                  c=colors[phase_name], s=150, marker='*', zorder=5)
        
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Height (Normalized)')
        ax2.set_title('Wrist Height Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 腰の回転角（簡易版）
        ax3 = axes[1, 0]
        hip_angles = []
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        for kp in keypoints:
            landmarks = kp.get('landmarks', {})
            left_shoulder = landmarks.get(LEFT_SHOULDER)
            right_shoulder = landmarks.get(RIGHT_SHOULDER)
            left_hip = landmarks.get(LEFT_HIP)
            right_hip = landmarks.get(RIGHT_HIP)
            
            if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                hip_center_y = (left_hip['y'] + right_hip['y']) / 2
                
                angle = np.arctan2(
                    shoulder_center_y - hip_center_y,
                    shoulder_center_x - hip_center_x
                )
                hip_angles.append(np.degrees(angle))
            else:
                hip_angles.append(np.nan)
        
        ax3.plot(frames, hip_angles, 'm-', linewidth=2, label='Hip Rotation Angle')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Angle (degrees)')
        ax3.set_title('Hip Rotation Angle Change')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. スイング速度（手首の速度）
        ax4 = axes[1, 1]
        velocities = []
        for i in range(len(wrist_heights)):
            if i == 0:
                velocities.append(0.0)
            else:
                if not (np.isnan(wrist_heights[i]) or np.isnan(wrist_heights[i-1])):
                    velocity = abs(wrist_heights[i] - wrist_heights[i-1])
                    velocities.append(velocity)
                else:
                    velocities.append(0.0)
        
        ax4.plot(frames, velocities, 'c-', linewidth=2, label='Wrist Velocity')
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('Velocity (Normalized)')
        ax4.set_title('Wrist Velocity Change')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"軌道グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_swings_side_by_side(
        self,
        my_keypoints: List[Dict],
        pro_keypoints: List[Dict],
        my_phases: Optional[Dict[str, int]] = None,
        pro_phases: Optional[Dict[str, int]] = None,
        save_path: Optional[str | Path] = None
    ):
        """
        2つのスイングを並べて比較
        
        Args:
            my_keypoints: 自分のスイングの骨格情報
            pro_keypoints: プロのスイングの骨格情報
            my_phases: 自分のスイング位相（オプション）
            pro_phases: プロのスイング位相（オプション）
            save_path: 保存パス（オプション）
        """
        if not my_keypoints or not pro_keypoints:
            print("警告: キーポイントが不足しています")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Swing Comparison: My Swing vs Pro', fontsize=16, fontweight='bold')
        
        # 手首の軌道を抽出
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        def extract_wrist_trajectory(keypoints):
            trajectory = []
            heights = []
            for kp in keypoints:
                landmarks = kp.get('landmarks', {})
                left_wrist = landmarks.get(LEFT_WRIST)
                right_wrist = landmarks.get(RIGHT_WRIST)
                
                if left_wrist and right_wrist:
                    center = (
                        (left_wrist['x'] + right_wrist['x']) / 2,
                        (left_wrist['y'] + right_wrist['y']) / 2
                    )
                    trajectory.append(center)
                    heights.append(center[1])
                elif left_wrist:
                    trajectory.append((left_wrist['x'], left_wrist['y']))
                    heights.append(left_wrist['y'])
                elif right_wrist:
                    trajectory.append((right_wrist['x'], right_wrist['y']))
                    heights.append(right_wrist['y'])
                else:
                    trajectory.append(None)
                    heights.append(np.nan)
            
            return trajectory, heights
        
        my_trajectory, my_heights = extract_wrist_trajectory(my_keypoints)
        pro_trajectory, pro_heights = extract_wrist_trajectory(pro_keypoints)
        
        # 1. 手首軌道の比較（2D）
        ax1 = axes[0, 0]
        if my_trajectory:
            my_x, my_y = zip(*[t for t in my_trajectory if t is not None])
            ax1.plot(my_x, my_y, 'b-', label='My Swing', linewidth=2, alpha=0.7)
        
        if pro_trajectory:
            pro_x, pro_y = zip(*[t for t in pro_trajectory if t is not None])
            ax1.plot(pro_x, pro_y, 'r-', label='Pro Swing', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('X Coordinate (Normalized)')
        ax1.set_ylabel('Y Coordinate (Normalized)')
        ax1.set_title('Wrist Trajectory Comparison (2D)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. 手首の高さの比較
        ax2 = axes[0, 1]
        my_frames = np.arange(len(my_heights))
        pro_frames = np.arange(len(pro_heights))
        
        ax2.plot(my_frames, my_heights, 'b-', label='My Swing', linewidth=2, alpha=0.7)
        ax2.plot(pro_frames, pro_heights, 'r-', label='Pro Swing', linewidth=2, alpha=0.7)
        
        # 位相をマーク
        if my_phases:
            colors = {'address': 'blue', 'top': 'red', 'impact': 'orange', 'finish': 'purple'}
            for phase_name, frame_num in my_phases.items():
                if 0 <= frame_num < len(my_heights):
                    ax2.axvline(x=frame_num, color=colors[phase_name], 
                              linestyle='--', linewidth=1, alpha=0.5)
        
        if pro_phases:
            colors = {'address': 'cyan', 'top': 'magenta', 'impact': 'yellow', 'finish': 'green'}
            for phase_name, frame_num in pro_phases.items():
                if 0 <= frame_num < len(pro_heights):
                    ax2.axvline(x=frame_num, color=colors[phase_name], 
                              linestyle=':', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Height (Normalized)')
        ax2.set_title('Wrist Height Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Xファクターの比較
        ax3 = axes[1, 0]
        
        def calculate_x_factor(keypoints):
            x_factors = []
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            
            for kp in keypoints:
                landmarks = kp.get('landmarks', {})
                left_shoulder = landmarks.get(LEFT_SHOULDER)
                right_shoulder = landmarks.get(RIGHT_SHOULDER)
                left_hip = landmarks.get(LEFT_HIP)
                right_hip = landmarks.get(RIGHT_HIP)
                
                if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                    shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                    shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                    hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                    hip_center_y = (left_hip['y'] + right_hip['y']) / 2
                    
                    shoulder_angle = np.arctan2(
                        shoulder_center_y - hip_center_y,
                        shoulder_center_x - hip_center_x
                    )
                    hip_angle = np.arctan2(
                        hip_center_y - shoulder_center_y,
                        hip_center_x - shoulder_center_x
                    )
                    
                    x_factor = abs(np.degrees(shoulder_angle - hip_angle))
                    x_factors.append(x_factor)
                else:
                    x_factors.append(np.nan)
            
            return x_factors
        
        my_x_factors = calculate_x_factor(my_keypoints)
        pro_x_factors = calculate_x_factor(pro_keypoints)
        
        ax3.plot(my_frames, my_x_factors, 'b-', label='My X-Factor', linewidth=2, alpha=0.7)
        ax3.plot(pro_frames, pro_x_factors, 'r-', label='Pro X-Factor', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('X-Factor (degrees)')
        ax3.set_title('X-Factor Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 位相ごとの比較
        ax4 = axes[1, 1]
        
        if my_phases and pro_phases:
            phase_names = ['address', 'top', 'impact', 'finish']
            my_values = []
            pro_values = []
            
            for phase_name in phase_names:
                my_frame = my_phases.get(phase_name, 0)
                pro_frame = pro_phases.get(phase_name, 0)
                
                if 0 <= my_frame < len(my_heights):
                    my_values.append(my_heights[my_frame] if not np.isnan(my_heights[my_frame]) else 0)
                else:
                    my_values.append(0)
                
                if 0 <= pro_frame < len(pro_heights):
                    pro_values.append(pro_heights[pro_frame] if not np.isnan(pro_heights[pro_frame]) else 0)
                else:
                    pro_values.append(0)
            
            x_pos = np.arange(len(phase_names))
            width = 0.35
            
            ax4.bar(x_pos - width/2, my_values, width, label='My Swing', alpha=0.7, color='blue')
            ax4.bar(x_pos + width/2, pro_values, width, label='Pro Swing', alpha=0.7, color='red')
            
            ax4.set_xlabel('Swing Phase')
            ax4.set_ylabel('Wrist Height (Normalized)')
            ax4.set_title('Wrist Height by Phase')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([p.capitalize() for p in phase_names])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Phase data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Phase Comparison')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"比較グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()

