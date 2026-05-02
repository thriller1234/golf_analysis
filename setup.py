"""
セットアップスクリプト
"""

from setuptools import setup, find_packages
from pathlib import Path

# READMEを読み込む
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="golf-swing-analysis",
    version="0.1.0",
    description="ゴルフスイング解析アプリ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/golf_analysis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.9.0",
        "mediapipe>=0.10.0",
        "numpy>=1.26.0",
        "ultralytics>=8.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)

