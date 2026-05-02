# ゴルフスイング解析アプリ

後方から撮影したスイング動画に **MediaPipe Pose（heavy）** の骨格オーバーレイを付与し、`output` に mp4 として書き出すツールです。比較指定時は左右横並びの1ファイルを出力します。

## プロジェクト概要

### 主な機能

- **骨格推定**: MediaPipe **Pose Landmarker heavy** + 動画用 `detect_for_video`（TFLite **GPU デリゲート**を優先、不可なら CPU）
- **出力**: 入力動画に骨格を重ねた mp4 を `output/` に保存。`--proswing` 指定時は myswing / proswing を横並びにした1本の mp4
- **クラブ位置**: 現状は未実装（将来 YOLOv8 等を検証する場合は `src/club_detection/` を参照）

## 環境要件

- **OS**: Windows 11
- **GPU**: GeForce RTX 4090 Laptop 16GB（MediaPipe は TensorFlow Lite GPU デリゲート利用可。環境により CPU フォールバック）
- **CUDA**: 12.8
- **Python**: 3.13.3

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd golf_analysis
```

### 2. 仮想環境の作成（推奨）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. 依存関係のインストール

```powershell
pip install -r requirements.txt
```

**注意**: PyTorchのCUDA 12.8対応版をインストールする場合、[PyTorch公式サイト](https://pytorch.org/)から適切なコマンドを確認してください。

例：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. 動作確認

```powershell
python -m pytest tests/
```

## プロジェクト構造

```
golf_analysis/
├── scripts/
│   └── main.py            # CLI（骨格オーバーレイ動画の生成）
├── src/
│   ├── pose_estimation/   # MediaPipe Pose Landmarker
│   ├── video_render.py    # オーバーレイ・横並び書き出し
│   ├── club_detection/    # クラブ検出モジュール（現状 main パイプライン未接続）
│   └── visualization/     # 骨格の OpenCV 描画
├── data/
│   ├── videos/
│   └── models/            # pose_landmarker_heavy.task（初回自動ダウンロード）
├── output/                # 生成 mp4
├── docs/
├── requirements.txt
└── README.md
```

## 使用方法

パスは **プロジェクトルートからの相対パス**、または **絶対パス**で指定します。

```powershell
# プロジェクトルートで実行（1本の動画 → output\<名前>_pose_overlay.mp4）
python scripts/main.py --myswing data/videos/my_swing.mp4

# 横並び比較 → output/compare_<左>_vs_<右>.mp4
python scripts/main.py --myswing data/videos/my_swing.mp4 --proswing data/videos/matsuyama_driver_1.mp4
```

初回実行時、`data/models/pose_landmarker_heavy.task` が無ければ自動ダウンロードします。

## 開発メモ（現状の焦点）

- MediaPipe Pose heavy + 動画オーバーレイ出力（`scripts/main.py`）
- クラブを動画に重ねる場合は `src/club_detection/` と `scripts/train_yolov8_club.py` をパイプラインに接続する必要あり（未接続）

## 注意事項

### 動画の要件

- **フレームレート**: 120fps以上を推奨（最低60fps）
- **解像度**: 1080p以上を推奨
- **撮影角度**: 後方（Down-the-Line）から撮影

### 精度について

このアプリケーションは以下の用途に適しています：
- ✅ フォーム改善・自己解析
- ✅ プロとの差分可視化
- ✅ スイング改善に十分な精度

以下の用途には適していません：
- ❌ トラックマン代替
- ❌ PGA計測機級の精度

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。詳細は[LICENSE](./LICENSE)ファイルをご覧ください。

## 貢献

プルリクエストやイシューの報告を歓迎します。
