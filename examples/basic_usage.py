"""
基本的な使用例
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.utils.video_utils import VideoUtils


def example_pose_estimation():
    """姿勢推定の使用例"""
    print("=== 姿勢推定の使用例 ===\n")
    
    # 動画ファイルのパス（実際のファイルに置き換えてください）
    video_path = project_root / "data" / "videos" / "my_swing.mp4"
    
    if not video_path.exists():
        print(f"動画ファイルが見つかりません: {video_path}")
        print("data/videos/ ディレクトリに動画ファイルを配置してください。")
        return
    
    # 動画情報を取得
    print("1. 動画情報を取得中...")
    video_info = VideoUtils.get_video_info(video_path)
    print(f"   解像度: {video_info['width']}x{video_info['height']}")
    print(f"   フレームレート: {video_info['fps']:.2f} fps")
    print(f"   総フレーム数: {video_info['frame_count']}")
    print(f"   動画の長さ: {video_info['duration']:.2f} 秒")
    
    # 姿勢推定を実行
    print("\n2. 姿勢推定を実行中...")
    estimator = PoseEstimator(model_complexity=2)
    keypoints = estimator.process_video(video_path)
    
    print(f"   骨格情報を {len(keypoints)} フレーム分取得しました")
    
    if keypoints:
        # 最初のフレームの主要関節を表示
        print("\n3. 最初のフレームの主要関節:")
        first_frame = keypoints[0]
        key_joints = estimator.get_key_joints(first_frame['landmarks'])
        for joint_name, (x, y) in key_joints.items():
            print(f"   {joint_name}: ({x:.3f}, {y:.3f})")
    
    print("\n完了しました！")


if __name__ == "__main__":
    example_pose_estimation()

