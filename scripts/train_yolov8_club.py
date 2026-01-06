"""
YOLOv8を使用したゴルフクラブ検出モデルの学習スクリプト
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil


def train_club_detector(
    dataset_path: str = "data/datasets/golf_club",
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: int = 0
):
    """
    ゴルフクラブ検出モデルの学習
    
    Args:
        dataset_path: データセットのパス
        model_size: モデルサイズ（n=軽量, s=小, m=中, l=大, x=超大）
        epochs: エポック数
        batch_size: バッチサイズ
        img_size: 画像サイズ
        device: GPUデバイスID（0がデフォルト）
    """
    dataset_path = Path(dataset_path)
    
    # データセットの存在確認
    if not dataset_path.exists():
        print(f"エラー: データセットが見つかりません: {dataset_path}")
        print("データセットを準備してください。詳細は docs/yolov8_custom_model_guide.md を参照してください。")
        return None
    
    # データセット設定ファイル（data.yaml）を作成
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # クラス数（ゴルフクラブのみ）
        'names': ['golf_club']
    }
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    
    print(f"データセット設定ファイルを作成しました: {yaml_path}")
    
    # モデルの初期化
    model_name = f'yolov8{model_size}.pt'
    print(f"モデルを初期化: {model_name}")
    model = YOLO(model_name)
    
    # 学習パラメータ
    print(f"\n学習を開始します...")
    print(f"  エポック数: {epochs}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  画像サイズ: {img_size}")
    print(f"  デバイス: GPU {device}")
    
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            workers=4,
            project='runs/detect',
            name='golf_club_detector',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # 最良のモデルを保存
        best_model_path = Path('runs/detect/golf_club_detector/weights/best.pt')
        target_path = Path('data/models/golf_club_detector.pt')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if best_model_path.exists():
            shutil.copy(best_model_path, target_path)
            print(f"\n✅ モデルを保存しました: {target_path}")
        else:
            print(f"\n⚠️ 警告: 最良のモデルが見つかりませんでした: {best_model_path}")
        
        # 学習結果のサマリーを表示
        print("\n学習結果:")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8ゴルフクラブ検出モデルの学習')
    parser.add_argument('--dataset', type=str, default='data/datasets/golf_club',
                       help='データセットのパス')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='モデルサイズ (n=軽量, s=小, m=中, l=大, x=超大)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='エポック数')
    parser.add_argument('--batch', type=int, default=16,
                       help='バッチサイズ')
    parser.add_argument('--img-size', type=int, default=640,
                       help='画像サイズ')
    parser.add_argument('--device', type=int, default=0,
                       help='GPUデバイスID')
    
    args = parser.parse_args()
    
    train_club_detector(
        dataset_path=args.dataset,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )

