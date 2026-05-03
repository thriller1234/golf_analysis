"""
YOLOv8 Instance Segmentation でゴルフクラブを学習し、
data/models/yolov8_club_seg.pt に best 重みをコピーする。

データセット: ポリゴンラベル付き YOLO segment 形式（train/images, train/labels, data.yaml）。

学習ログ・曲線・weights は既定で output/train/<run_name>/ に保存（旧 Ultralytics 既定の runs/ ではない）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def train_club_segmenter(
    dataset_path: str | Path = "data/datasets/golf_club_seg",
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: int | str = 0,
    project: str = "output/train",
    run_name: str = "golf_club_seg",
    *,
    plots: bool = False,
) -> Path | None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"エラー: データセットが見つかりません: {dataset_path}")
        print("ポリゴン付き YOLO segment データを準備してください（train/images, train/labels）。")
        return None

    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        data_yaml = {
            "path": str(dataset_path.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": 1,
            "names": ["golf_club"],
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True)
        print(f"data.yaml を作成しました: {data_yaml_path}")

    base = f"yolov8{model_size}-seg.pt"
    print(f"初期モデル: {base}")
    model = YOLO(base)

    print("\nセグメンテーション学習を開始します…")
    print(f"  dataset: {data_yaml_path}")
    print(f"  epochs={epochs}, batch={batch_size}, imgsz={img_size}, device={device}, plots={plots}")
    if not plots:
        print("  （画像・曲線 PNG は最小限。PR 曲線や train/val バッチ画像も出したい場合は --plots）")

    try:
        model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            workers=4,
            project=project,
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            plots=plots,
        )

        best_src = Path(project) / run_name / "weights" / "best.pt"
        target = Path("data/models/yolov8_club_seg.pt")
        target.parent.mkdir(parents=True, exist_ok=True)

        if best_src.exists():
            shutil.copy(best_src, target)
            print(f"\n✅ モデルをコピーしました: {target.resolve()}")
            return target
        print(f"\n⚠️ best.pt が見つかりません: {best_src}")
        return None
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLOv8 Segment — ゴルフクラブ")
    p.add_argument("--dataset", type=str, default="data/datasets/golf_club_seg")
    p.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--project",
        type=str,
        default="output/train",
        help="Ultralytics の project（学習成果物の親ディレクトリ）",
    )
    p.add_argument(
        "--name",
        type=str,
        default="golf_club_seg",
        help="Ultralytics の run 名（output/train/<name>/ に保存）",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="PR 曲線・混同行列・train/val バッチ画像などを追加保存（ファイル数が増えます）",
    )
    args = p.parse_args()

    train_club_segmenter(
        dataset_path=args.dataset,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        run_name=args.name,
        plots=args.plots,
    )
