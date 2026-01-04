"""
サイドバイサイド比較可視化のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.swing_analysis import PhaseDetector
from src.visualization import SwingVisualizer


def test_side_by_side():
    """サイドバイサイド比較可視化のテスト"""
    print("=== サイドバイサイド比較可視化のテスト ===\n")
    
    # 動画ファイルのパス
    my_swing_path = project_root / "data" / "videos" / "my_swing.mp4"
    pro_swing_path = project_root / "data" / "videos" / "matsuyama_iron_1.mp4"
    
    if not my_swing_path.exists():
        print(f"自分のスイング動画が見つかりません: {my_swing_path}")
        return
    
    if not pro_swing_path.exists():
        print(f"プロ選手のスイング動画が見つかりません: {pro_swing_path}")
        return
    
    # 1. 姿勢推定
    print("1. 姿勢推定を実行中...")
    estimator = PoseEstimator()
    my_keypoints = estimator.process_video(my_swing_path)
    pro_keypoints = estimator.process_video(pro_swing_path)
    print(f"   自分のスイング: {len(my_keypoints)} フレーム")
    print(f"   プロ選手のスイング: {len(pro_keypoints)} フレーム")
    
    # 2. 位相検出
    print("\n2. スイング位相を検出中...")
    phase_detector = PhaseDetector()
    my_phases = phase_detector.detect_phases(my_keypoints)
    pro_phases = phase_detector.detect_phases(pro_keypoints)
    print(f"   自分の位相: {my_phases}")
    print(f"   プロ選手の位相: {pro_phases}")
    
    # 3. サイドバイサイド比較
    print("\n3. サイドバイサイド比較を生成中...")
    visualizer = SwingVisualizer()
    output_path = project_root / "output" / "side_by_side_comparison.png"
    visualizer.compare_swings_side_by_side(
        my_keypoints,
        pro_keypoints,
        my_phases=my_phases,
        pro_phases=pro_phases,
        save_path=output_path
    )
    print(f"   比較グラフを保存しました: {output_path}")
    
    print("\n完了しました！")


if __name__ == "__main__":
    test_side_by_side()

