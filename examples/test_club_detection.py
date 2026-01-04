"""
クラブ検出機能のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.club_detection import ClubDetector


def test_club_detection():
    """クラブ検出機能のテスト"""
    print("=== クラブ検出機能のテスト ===\n")
    
    # 動画ファイルのパス
    video_path = project_root / "data" / "videos" / "my_swing.mp4"
    
    if not video_path.exists():
        print(f"動画ファイルが見つかりません: {video_path}")
        return
    
    # 1. 姿勢推定（手首位置を取得するため）
    print("1. 姿勢推定を実行中...")
    estimator = PoseEstimator()
    keypoints = estimator.process_video(video_path)
    print(f"   骨格情報を {len(keypoints)} フレーム分取得しました")
    
    # 2. クラブ検出
    print("\n2. クラブ検出を実行中...")
    try:
        # YOLOv8を使用する場合
        detector = ClubDetector(use_yolo=True)
        print("   YOLOv8モードで検出中...")
    except Exception as e:
        print(f"   YOLOv8の使用に失敗: {e}")
        print("   簡易検出モード（手首位置ベース）を使用します...")
        detector = ClubDetector(use_yolo=False)
    
    club_detections = detector.detect_club_in_video(video_path, keypoints)
    print(f"   クラブ検出を {len(club_detections)} フレーム分実行しました")
    
    # 3. クラブ軌道の抽出
    print("\n3. クラブ軌道を抽出中...")
    trajectory = detector.extract_club_trajectory(club_detections)
    
    print(f"   クラブヘッド軌道: {len(trajectory['club_head'])} ポイント")
    print(f"   グリップ軌道: {len(trajectory['grip'])} ポイント")
    
    if trajectory['club_head']:
        print("\n   クラブヘッド軌道のサンプル（最初の5ポイント）:")
        for i, point in enumerate(trajectory['club_head'][:5]):
            print(f"     フレーム {point['frame']}: 位置={point['position']}, "
                  f"信頼度={point.get('confidence', 0.0):.2f}")
    
    # 4. 手首位置ベースの簡易検出も試す
    print("\n4. 手首位置ベースの簡易検出を実行中...")
    wrist_trajectory = detector.detect_club_from_wrist(keypoints)
    print(f"   手首ベース軌道: ヘッド={len(wrist_trajectory['club_head'])} ポイント, "
          f"グリップ={len(wrist_trajectory['grip'])} ポイント")
    
    print("\n完了しました！")


if __name__ == "__main__":
    test_club_detection()

