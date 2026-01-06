"""
エッジ検出統合のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.club_detection import ClubDetector


def test_edge_detection_integration():
    """エッジ検出統合のテスト"""
    print("=== エッジ検出統合のテスト ===\n")
    
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
    
    # 2. エッジ検出を有効にしたクラブ検出
    print("\n2. エッジ検出統合クラブ検出を実行中...")
    
    # 方法1: YOLOv8 + エッジ検出の組み合わせ
    try:
        detector_combined = ClubDetector(use_yolo=True, use_edge_detection=True)
        print("   YOLOv8 + エッジ検出モードで検出中...")
        club_detections_combined = detector_combined.detect_club_in_video(
            video_path, 
            keypoints,
            combine_methods=True
        )
        print(f"   検出完了: {len(club_detections_combined)} フレーム")
        
        # 検出方法の統計
        methods = {}
        for frame_data in club_detections_combined:
            for det in frame_data.get('detections', []):
                method = det.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
        
        print(f"   検出方法の統計: {methods}")
        
    except Exception as e:
        print(f"   YOLOv8の使用に失敗: {e}")
    
    # 方法2: エッジ検出のみ
    print("\n3. エッジ検出のみモードで検出中...")
    detector_edge_only = ClubDetector(use_yolo=False, use_edge_detection=True)
    club_detections_edge = detector_edge_only.detect_club_in_video(
        video_path,
        keypoints,
        combine_methods=False
    )
    print(f"   検出完了: {len(club_detections_edge)} フレーム")
    
    # 軌道の抽出
    print("\n4. クラブ軌道を抽出中...")
    trajectory = detector_edge_only.extract_club_trajectory(club_detections_edge)
    
    print(f"   クラブヘッド軌道: {len(trajectory['club_head'])} ポイント")
    print(f"   グリップ軌道: {len(trajectory['grip'])} ポイント")
    
    if trajectory['club_head']:
        print("\n   クラブヘッド軌道のサンプル（最初の5ポイント）:")
        for i, point in enumerate(trajectory['club_head'][:5]):
            print(f"     フレーム {point['frame']}: 位置={point['position']}, "
                  f"信頼度={point.get('confidence', 0.0):.2f}")
    
    print("\n完了しました！")


if __name__ == "__main__":
    test_edge_detection_integration()

