# ゴルフスイング解析アプリ

後方から撮影した1つのカメラ動画を使用して、自分のスイングとプロ選手のスイングを比較解析するアプリケーションです。

## プロジェクト概要

このプロジェクトは、AI技術を活用してゴルフスイングを定量的に解析し、プロ選手との比較を行うことで、スイング改善を支援します。

### 主な機能

- **骨格推定**: MediaPipe/OpenPoseを使用した人体骨格の検出
- **クラブ検出**: YOLOv8を使用したゴルフクラブの検出とトラッキング
- **スイング位相検出**: アドレス、トップ、インパクト、フィニッシュの自動検出
- **比較解析**: プロ選手とのスイング比較と差分可視化
- **可視化**: スイング軌道、角度、タイミングの可視化

## 環境要件

- **OS**: Windows 11
- **GPU**: GeForce RTX 5090 Laptop 16GB
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
├── src/                    # ソースコード
│   ├── pose_estimation/   # 骨格推定モジュール
│   ├── club_detection/    # クラブ検出モジュール
│   ├── swing_analysis/    # スイング解析モジュール
│   ├── visualization/     # 可視化モジュール
│   └── utils/             # ユーティリティ
├── data/                  # データディレクトリ
│   ├── videos/            # 動画ファイル
│   └── models/            # 学習済みモデル
├── tests/                 # テストコード
├── docs/                  # ドキュメント
│   └── howtodo.md         # 開発ガイド
├── requirements.txt       # 依存関係
└── README.md             # このファイル
```

## 使用方法

### STEP 1: 姿勢推定による骨格抽出

```python
from src.pose_estimation import PoseEstimator

estimator = PoseEstimator()
keypoints = estimator.process_video("data/videos/my_swing.mp4")
```

### STEP 2: クラブ検出

```python
from src.club_detection import ClubDetector

detector = ClubDetector()
club_trajectory = detector.detect_club("data/videos/my_swing.mp4")
```

### STEP 3: スイング解析と比較

```python
from src.swing_analysis import SwingAnalyzer

analyzer = SwingAnalyzer()
comparison = analyzer.compare_swings(
    my_swing="data/videos/my_swing.mp4",
    pro_swing="data/videos/pro_swing.mp4"
)
```

詳細な使用方法は `docs/howtodo.md` を参照してください。

## 開発ロードマップ

- [x] プロジェクト構造の作成
- [x] STEP 1: 姿勢推定による骨格抽出（完了）
  - [x] MediaPipe 0.10対応
  - [x] 動画からの骨格情報抽出
  - [x] 主要関節の座標取得
  - [x] 骨格情報の可視化（動画への描画）
- [x] STEP 2: クラブ検出モデル作成（基本実装完了）
  - [x] クラス構造の作成
  - [x] YOLOv8によるクラブ検出の実装（基本機能）
  - [x] クラブ軌道の抽出（基本実装）
  - [x] 手首位置ベースの簡易検出（フォールバック）
  - [ ] カスタムモデルの学習（ゴルフクラブ専用）
  - [ ] エッジ検出との組み合わせ（実装済みだが未統合）
- [x] STEP 3: スイング位相の自動検出と正規化（完了）
  - [x] クラス構造の作成
  - [x] アドレス、トップ、インパクト、フィニッシュの自動検出
  - [x] 空間・時間の正規化実装（肩幅・腰幅ベースの空間正規化、位相ベースの時間正規化）
- [x] STEP 4: 比較解析と可視化（完了）
  - [x] クラス構造の作成
  - [x] プロ選手との比較機能
  - [x] スイング軌道の可視化
  - [x] 特徴量の抽出と比較（肩・腰の回転角、Xファクター、手首軌道）
  - [x] サイドバイサイド比較可視化
- [ ] STEP 5: UIの実装

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

[ライセンス情報を追加]

## 参考資料

詳細な技術情報は `docs/howtodo.md` を参照してください。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 連絡先

[連絡先情報を追加]

