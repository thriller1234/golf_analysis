"""
可視化機能のテスト
"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.swing_analysis import PhaseDetector
from src.visualization import SwingVisualizer
import cv2


def test_visualization():
    """可視化機能のテスト"""
    print("=== 可視化機能のテスト ===\n")
    
    # 動画ファイルのパス
    video_path = project_root / "data" / "videos" / "my_swing.mp4"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
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
    print(f"   アドレス: {phases['address']}, トップ: {phases['top']}, "
          f"インパクト: {phases['impact']}, フィニッシュ: {phases['finish']}")
    
    # 3. 骨格を描画した動画を作成
    print("\n3. 骨格を描画した動画を作成中...")
    visualizer = SwingVisualizer()
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = output_dir / "swing_with_skeleton.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(keypoints):
            kp = keypoints[frame_idx]
            annotated_frame = visualizer.visualize_keypoints(frame, kp)
            
            # 位相をテキストで表示（インパクトは検出精度が低いため、注意書き付き）
            phase_text = ""
            if frame_idx == phases['address']:
                phase_text = "ADDRESS"
            elif frame_idx == phases['top']:
                phase_text = "TOP"
            elif frame_idx == phases['impact']:
                # インパクト検出は現時点では近似値（クラブ検出により精度向上可能）
                phase_text = "IMPACT (approx)"
            elif frame_idx == phases['finish']:
                phase_text = "FINISH"
            
            if phase_text:
                cv2.putText(annotated_frame, phase_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            out.write(annotated_frame)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"   動画を保存しました: {output_video_path}")
    
    # 4. スイング軌道をプロット
    print("\n4. スイング軌道をプロット中...")
    trajectory_path = output_dir / "swing_trajectory.png"
    visualizer.plot_swing_trajectory(keypoints, phases, trajectory_path)
    print(f"   軌道グラフを保存しました: {trajectory_path}")
    
    print("\n完了しました！")
    print(f"\n出力ファイル:")
    print(f"  - {output_video_path}")
    print(f"  - {trajectory_path}")


if __name__ == "__main__":
    test_visualization()

