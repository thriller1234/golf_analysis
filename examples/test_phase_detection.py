"""
スイング位相検出のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.swing_analysis import PhaseDetector


def test_phase_detection():
    """スイング位相検出のテスト"""
    print("=== スイング位相検出のテスト ===\n")
    
    # 動画ファイルのパス
    video_path = project_root / "data" / "videos" / "my_swing.mp4"
    
    if not video_path.exists():
        print(f"動画ファイルが見つかりません: {video_path}")
        return
    
    # 1. 姿勢推定
    print("1. 姿勢推定を実行中...")
    estimator = PoseEstimator()
    keypoints = estimator.process_video(video_path)
    print(f"   骨格情報を {len(keypoints)} フレーム分取得しました")
    
    # 2. 位相検出
    print("\n2. スイング位相を検出中...")
    phase_detector = PhaseDetector()
    phases = phase_detector.detect_phases(keypoints)
    
    print("\n検出された位相:")
    print(f"   アドレス: フレーム {phases['address']}")
    print(f"   トップ:   フレーム {phases['top']}")
    print(f"   インパクト: フレーム {phases['impact']}")
    print(f"   フィニッシュ: フレーム {phases['finish']}")
    
    # 3. 詳細情報
    print("\n3. 詳細情報:")
    total_frames = len(keypoints)
    print(f"   総フレーム数: {total_frames}")
    print(f"   アドレス → トップ: {phases['top'] - phases['address']} フレーム")
    print(f"   トップ → インパクト: {phases['impact'] - phases['top']} フレーム")
    print(f"   インパクト → フィニッシュ: {phases['finish'] - phases['impact']} フレーム")
    
    # 手首の高さの変化を確認
    print("\n4. 手首の高さの変化:")
    heights = phase_detector._calculate_wrist_heights(keypoints)
    if len(heights) > 0:
        print(f"   アドレス時の高さ: {heights[phases['address']]:.3f}")
        print(f"   トップ時の高さ: {heights[phases['top']]:.3f}")
        print(f"   インパクト時の高さ: {heights[phases['impact']]:.3f}")
        print(f"   フィニッシュ時の高さ: {heights[phases['finish']]:.3f}")
    
    print("\n完了しました！")


if __name__ == "__main__":
    test_phase_detection()

