"""
クラブ検出モジュール
YOLOv8を使用したゴルフクラブの検出とトラッキング
"""

from .club_detector import ClubDetector
from .club_detector_enhanced import EnhancedClubDetector

__all__ = ["ClubDetector", "EnhancedClubDetector"]

