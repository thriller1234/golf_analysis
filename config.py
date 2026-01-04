"""
設定ファイル
"""

from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
MODELS_DIR = DATA_DIR / "models"

# 姿勢推定の設定
POSE_ESTIMATION = {
    "model_complexity": 2,  # 0, 1, 2 (2が最高精度)
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# クラブ検出の設定
CLUB_DETECTION = {
    "model_path": None,  # カスタムモデルのパス（Noneの場合はデフォルト）
    "confidence_threshold": 0.5,
}

# スイング解析の設定
SWING_ANALYSIS = {
    "min_fps": 60,  # 最低フレームレート
    "recommended_fps": 120,  # 推奨フレームレート
}

# フィルタリングの設定
FILTERING = {
    "use_kalman_filter": True,
    "kalman_process_noise": 0.1,
    "kalman_measurement_noise": 1.0,
    "moving_average_window": 5,
}

# 出力設定
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

