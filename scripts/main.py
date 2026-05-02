"""
ゴルフスイング動画に骨格オーバーレイを描画し、output に mp4 を書き出す。

使用例（プロジェクトルートで）:
  python scripts/main.py --myswing data/videos/my_swing.mp4
  python scripts/main.py --myswing data/videos/my_swing.mp4 --proswing data/videos/pro.mp4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# パッケージ src を解決するためリポジトリルートを追加
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pose_estimation import PoseEstimator
from src.video_render import render_pose_overlay_video, render_side_by_side_overlay
from src.visualization import SwingVisualizer


def _project_root() -> Path:
    return _ROOT


def _resolve_video_path(project_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def _safe_filename_segment(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="スイング動画へ骨格オーバーレイを付与して output に保存します。"
    )
    parser.add_argument(
        "--myswing",
        type=str,
        required=True,
        help="プロジェクトルートからの相対パス、または絶対パス（自分のスイング動画）",
    )
    parser.add_argument(
        "--proswing",
        type=str,
        default=None,
        help="プロジェクトルートからの相対パス、または絶対パス（比較用プロの動画）",
    )

    args = parser.parse_args()
    root = _project_root()
    my_path = _resolve_video_path(root, args.myswing)

    if not my_path.exists():
        print(f"エラー: 動画が見つかりません: {my_path}", file=sys.stderr)
        return 1

    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    visualizer = SwingVisualizer()

    if args.proswing:
        pro_path = _resolve_video_path(root, args.proswing)
        if not pro_path.exists():
            print(f"エラー: 動画が見つかりません: {pro_path}", file=sys.stderr)
            return 1

        my_stem = _safe_filename_segment(my_path.stem)
        pro_stem = _safe_filename_segment(pro_path.stem)
        output_path = out_dir / f"compare_{my_stem}_vs_{pro_stem}.mp4"

        print("Pose Landmarker heavy（GPU 利用可）で左右それぞれ推定します…")
        est_my = PoseEstimator()
        est_pro = PoseEstimator()
        try:
            render_side_by_side_overlay(
                my_path,
                pro_path,
                output_path,
                est_my,
                est_pro,
                visualizer,
                label_my="myswing",
                label_pro="proswing",
            )
        finally:
            est_my.close()
            est_pro.close()

        print(f"出力しました: {output_path}")
        return 0

    output_path = out_dir / f"{_safe_filename_segment(my_path.stem)}_pose_overlay.mp4"

    print("Pose Landmarker heavy（GPU 利用可）で姿勢推定しています…")
    estimator = PoseEstimator()
    try:
        render_pose_overlay_video(
            my_path,
            output_path,
            estimator,
            visualizer,
            label=None,
        )
    finally:
        estimator.close()

    print(f"出力しました: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
