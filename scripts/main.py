"""
ゴルフスイング動画に骨格（MediaPipe）＋クラブ（YOLOv8 Segment）を重ね、output/videos に mp4 を書き出す。

使用例（プロジェクトルートで）:
  python scripts/main.py --myswing data/videos/my_swing.mp4
  python scripts/main.py --myswing data/videos/my_swing.mp4 --proswing data/videos/pro.mp4
  python scripts/main.py --myswing data/videos/my.mp4 --no-club   # 姿勢のみ（モデル不要）
  python scripts/main.py --myswing data/videos/my.mp4 --no-off-line-only  # 不足時のみネット取得
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.club_detection import ClubSegmentor
from src.pose_estimation import PoseEstimator
from src.video_render import render_pose_overlay_video, render_side_by_side_overlay
from src.visualization import SwingVisualizer


DEFAULT_CLUB_MODEL = Path("data/models/yolov8_club_seg.pt")
# Ultralytics release asset (pretrained seg; not club-finetuned — see README)
_YOLOV8N_SEG_PT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"
)


def _project_root() -> Path:
    return _ROOT


def _resolve_path(project_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def _safe_filename_segment(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"ダウンロード中: {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, str(dest))
    print(f"保存しました: {dest}", file=sys.stderr)


def _resolve_club_model(
    root: Path,
    club_model_arg: str,
    offline_only: bool,
) -> Path | None:
    """クラブ用 .pt のパス。--no-club 時は呼ばない。"""
    path = _resolve_path(root, club_model_arg)
    if path.is_file():
        return path

    if offline_only:
        print(
            f"エラー: クラブ用モデルがありません（オフラインモード）: {path}\n"
            "  data/models に配置するか、学習して data/models/yolov8_club_seg.pt を用意してください。\n"
            "  ネットから不足分のみ取得する場合は --no-off-line-only を指定します（事前学習重みのみ自動取得可）。\n"
            "  骨格のみなら --no-club を使います。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    default_club = _resolve_path(root, str(DEFAULT_CLUB_MODEL))
    if path.resolve() == default_club.resolve():
        fallback = root / "data" / "models" / "yolov8n-seg.pt"
        if not fallback.is_file():
            print(
                "注意: data/models/yolov8_club_seg.pt がありません。"
                " 事前学習の yolov8n-seg.pt を取得します（クラブ特化の学習済みではありません）。",
                file=sys.stderr,
            )
            _download_file(_YOLOV8N_SEG_PT_URL, fallback)
        else:
            print(
                f"注意: {path.name} が無いため、既存の {fallback.name} を使います（クラブ特化学習済みではない場合があります）。",
                file=sys.stderr,
            )
        return fallback

    print(
        f"エラー: 指定パスにクラブ用 .pt がありません: {path}\n"
        "  ファイルを置くか、既定の data/models/yolov8_club_seg.pt を用意してください。",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _pose_estimator(offline_only: bool) -> PoseEstimator:
    try:
        return PoseEstimator(offline_only=offline_only)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="スイング動画へ骨格（MediaPipe）とクラブ（YOLOv8 Segment）を重ねて output/videos に保存します。"
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
        help="比較用プロの動画（任意）",
    )
    parser.add_argument(
        "--club-model",
        type=str,
        default=str(DEFAULT_CLUB_MODEL),
        help=f"学習済み YOLOv8 Segment の .pt（既定: {DEFAULT_CLUB_MODEL}）",
    )
    parser.add_argument(
        "--no-club",
        action="store_true",
        help="クラブ推論を行わず骨格のみ出力する（クラブ用 .pt 不要）",
    )
    parser.add_argument(
        "--club-samples",
        type=int,
        default=8,
        help="クラブマスク主軸に沿ったサンプル点数（既定: 8）",
    )
    parser.add_argument(
        "--off-line-only",
        dest="offline_only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "既定: data/models のみ使用（不足時はエラー）。"
            " --no-off-line-only で Pose の .task / 既定クラブ .pt 不足時にネットから取得を試みます。"
        ),
    )

    args = parser.parse_args()
    root = _project_root()
    my_path = _resolve_path(root, args.myswing)

    if not my_path.exists():
        print(f"エラー: 動画が見つかりません: {my_path}", file=sys.stderr)
        return 1

    out_dir = root / "output" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    visualizer = SwingVisualizer()

    club_seg: ClubSegmentor | None = None
    if not args.no_club:
        club_path = _resolve_club_model(root, args.club_model, args.offline_only)
        print(f"クラブ推論: YOLOv8 Segment ({club_path.name})")
        club_seg = ClubSegmentor(club_path)

    if args.proswing:
        pro_path = _resolve_path(root, args.proswing)
        if not pro_path.exists():
            print(f"エラー: 動画が見つかりません: {pro_path}", file=sys.stderr)
            return 1

        my_stem = _safe_filename_segment(my_path.stem)
        pro_stem = _safe_filename_segment(pro_path.stem)
        output_path = out_dir / f"compare_{my_stem}_vs_{pro_stem}_swing_overlay.mp4"

        print("Pose Landmarker heavy + クラブ（指定時）で左右それぞれ処理します…")
        est_my = _pose_estimator(args.offline_only)
        est_pro = _pose_estimator(args.offline_only)
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
                club_segmentor=club_seg,
                num_club_samples=args.club_samples,
            )
        finally:
            est_my.close()
            est_pro.close()

        print(f"出力しました: {output_path}")
        return 0

    output_path = out_dir / f"{_safe_filename_segment(my_path.stem)}_swing_overlay.mp4"

    print("Pose Landmarker heavy + クラブ（指定時）で処理します…")
    estimator = _pose_estimator(args.offline_only)
    try:
        render_pose_overlay_video(
            my_path,
            output_path,
            estimator,
            visualizer,
            label=None,
            club_segmentor=club_seg,
            num_club_samples=args.club_samples,
        )
    finally:
        estimator.close()

    print(f"出力しました: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
