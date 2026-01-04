"""
正規化機能のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.swing_analysis import PhaseDetector, SwingNormalizer


def test_normalization():
    """正規化機能のテスト"""
    print("=== 正規化機能のテスト ===\n")
    
    # 動画ファイルのパス
    my_swing_path = project_root / "data" / "videos" / "my_swing.mp4"
    pro_swing_path = project_root / "data" / "videos" / "matsuyama_iron_1.mp4"
    
    if not my_swing_path.exists():
        print(f"自分のスイング動画が見つかりません: {my_swing_path}")
        return
    
    # 1. 姿勢推定
    print("1. 姿勢推定を実行中...")
    estimator = PoseEstimator()
    my_keypoints = estimator.process_video(my_swing_path)
    print(f"   自分のスイング: {len(my_keypoints)} フレーム")
    
    if pro_swing_path.exists():
        pro_keypoints = estimator.process_video(pro_swing_path)
        print(f"   プロ選手のスイング: {len(pro_keypoints)} フレーム")
    else:
        print(f"   プロ選手のスイングが見つかりません: {pro_swing_path}")
        pro_keypoints = None
    
    # 2. 空間正規化
    print("\n2. 空間正規化を実行中...")
    normalizer = SwingNormalizer()
    
    # 肩幅を計算
    my_shoulder_width = normalizer._get_reference_scale(my_keypoints, use_shoulder=True)
    print(f"   自分の肩幅（平均）: {my_shoulder_width:.4f}")
    
    if pro_keypoints:
        pro_shoulder_width = normalizer._get_reference_scale(pro_keypoints, use_shoulder=True)
        print(f"   プロ選手の肩幅（平均）: {pro_shoulder_width:.4f}")
        print(f"   スケール比: {pro_shoulder_width / my_shoulder_width:.2f}")
        
        # プロ選手を基準に正規化
        normalized_my = normalizer.normalize_spatial(
            my_keypoints,
            reference_keypoints=pro_keypoints,
            use_shoulder_width=True
        )
        print(f"   正規化完了: {len(normalized_my)} フレーム")
    
    # 3. 時間正規化
    print("\n3. 時間正規化を実行中...")
    phase_detector = PhaseDetector()
    my_phases = phase_detector.detect_phases(my_keypoints)
    
    target_length = 100
    normalized_temporal = normalizer.normalize_temporal(
        my_keypoints,
        target_length,
        phases=my_phases
    )
    print(f"   元のフレーム数: {len(my_keypoints)}")
    print(f"   正規化後のフレーム数: {len(normalized_temporal)}")
    print(f"   目標フレーム数: {target_length}")
    
    # 4. 位相ベースの正規化
    if pro_keypoints:
        print("\n4. 位相ベースの時間正規化を実行中...")
        pro_phases = phase_detector.detect_phases(pro_keypoints)
        
        # プロ選手のフレーム数に合わせる
        normalized_by_phase = normalizer.normalize_temporal(
            my_keypoints,
            len(pro_keypoints),
            phases=my_phases
        )
        print(f"   プロ選手のフレーム数: {len(pro_keypoints)}")
        print(f"   正規化後のフレーム数: {len(normalized_by_phase)}")
    
    print("\n完了しました！")


if __name__ == "__main__":
    test_normalization()

