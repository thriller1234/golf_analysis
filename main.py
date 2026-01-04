"""
ゴルフスイング解析アプリ - メインスクリプト
"""

import argparse
from pathlib import Path
from src.pose_estimation import PoseEstimator
from src.club_detection import ClubDetector
from src.swing_analysis import SwingAnalyzer
from src.visualization import SwingVisualizer


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="ゴルフスイング解析アプリ"
    )
    parser.add_argument(
        "video",
        type=str,
        help="解析する動画ファイルのパス"
    )
    parser.add_argument(
        "--pro-video",
        type=str,
        help="比較するプロ選手の動画ファイルのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="出力ディレクトリ（デフォルト: output）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pose", "club", "analysis", "all"],
        default="all",
        help="実行モード（デフォルト: all）"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return
    
    print(f"動画を解析中: {video_path}")
    
    # 姿勢推定
    if args.mode in ["pose", "all"]:
        print("\n[1/4] 姿勢推定を実行中...")
        estimator = PoseEstimator()
        keypoints = estimator.process_video(video_path)
        print(f"  骨格情報を {len(keypoints)} フレーム分取得しました")
    
    # クラブ検出
    if args.mode in ["club", "all"]:
        print("\n[2/4] クラブ検出を実行中...")
        detector = ClubDetector()
        club_detections = detector.detect_club_in_video(video_path)
        print(f"  クラブ検出を {len(club_detections)} フレーム分実行しました")
    
    # スイング解析
    if args.mode in ["analysis", "all"]:
        print("\n[3/4] スイング解析を実行中...")
        analyzer = SwingAnalyzer()
        
        if args.pro_video:
            pro_video_path = Path(args.pro_video)
            if pro_video_path.exists():
                print(f"  プロ選手の動画と比較中: {pro_video_path}")
                comparison = analyzer.compare_swings(video_path, pro_video_path)
                print("  比較解析が完了しました")
            else:
                print(f"  警告: プロ選手の動画が見つかりません: {pro_video_path}")
        else:
            print("  プロ選手の動画が指定されていません（--pro-videoオプション）")
    
    # 可視化
    if args.mode == "all":
        print("\n[4/4] 可視化を実行中...")
        visualizer = SwingVisualizer()
        print("  可視化が完了しました")
    
    print("\n解析が完了しました！")


if __name__ == "__main__":
    main()

