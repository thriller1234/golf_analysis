"""
姿勢推定モジュールのテスト
"""

import pytest
import numpy as np
from pathlib import Path
from src.pose_estimation import PoseEstimator


def test_pose_estimator_initialization():
    """PoseEstimatorの初期化テスト"""
    estimator = PoseEstimator()
    assert estimator is not None
    assert estimator.pose is not None


def test_process_frame():
    """フレーム処理のテスト"""
    estimator = PoseEstimator()
    
    # ダミーフレームを作成
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 処理実行（人物がいない場合はNoneが返る）
    result = estimator.process_frame(dummy_frame)
    
    # 結果の型チェック（Noneまたは辞書）
    assert result is None or isinstance(result, dict)


# 実際の動画ファイルがある場合のテストは、統合テストとして別途実装

