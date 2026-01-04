"""
プロ選手との比較機能のテスト
"""

from pathlib import Path
import sys
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.swing_analysis import SwingAnalyzer


def test_comparison():
    """プロ選手との比較機能のテスト"""
    print("=== プロ選手との比較機能のテスト ===\n")
    
    # 動画ファイルのパス
    my_swing_path = project_root / "data" / "videos" / "my_swing.mp4"
    pro_swing_path = project_root / "data" / "videos" / "matsuyama_iron_1.mp4"
    
    if not my_swing_path.exists():
        print(f"自分のスイング動画が見つかりません: {my_swing_path}")
        return
    
    if not pro_swing_path.exists():
        print(f"プロ選手のスイング動画が見つかりません: {pro_swing_path}")
        return
    
    # 比較解析を実行
    print("1. スイング解析を実行中...")
    analyzer = SwingAnalyzer()
    result = analyzer.compare_swings(my_swing_path, pro_swing_path)
    
    print("\n2. 比較結果:")
    print(f"   自分のスイング: {result['my_swing']}")
    print(f"   プロ選手のスイング: {result['pro_swing']}")
    
    # 位相の比較
    print("\n3. 位相の比較:")
    my_phases = result['comparison']['my_phases']
    pro_phases = result['comparison']['pro_phases']
    
    phase_names = {'address': 'アドレス', 'top': 'トップ', 'impact': 'インパクト', 'finish': 'フィニッシュ'}
    for phase_name, phase_label in phase_names.items():
        my_frame = my_phases.get(phase_name, 'N/A')
        pro_frame = pro_phases.get(phase_name, 'N/A')
        print(f"   {phase_label}: 自分={my_frame}, プロ={pro_frame}")
    
    # 特徴量の比較
    print("\n4. 特徴量の比較:")
    phase_comparison = result['comparison']['phase_comparison']
    
    for phase_name, phase_label in phase_names.items():
        if phase_name in phase_comparison:
            comp = phase_comparison[phase_name]
            print(f"\n   {phase_label}:")
            
            if comp.get('shoulder_rotation_diff') is not None:
                print(f"     肩の回転角の差: {comp['shoulder_rotation_diff']:.2f}度")
            if comp.get('hip_rotation_diff') is not None:
                print(f"     腰の回転角の差: {comp['hip_rotation_diff']:.2f}度")
            if comp.get('x_factor_diff') is not None:
                print(f"     Xファクターの差: {comp['x_factor_diff']:.2f}度")
            if comp.get('wrist_height_diff') is not None:
                print(f"     手首の高さの差: {comp['wrist_height_diff']:.4f}")
    
    # 全体的な特徴量
    print("\n5. 全体的な特徴量:")
    feature_diffs = result['comparison']['feature_differences']
    if feature_diffs.get('max_x_factor_my') is not None:
        print(f"   最大Xファクター（自分）: {feature_diffs['max_x_factor_my']:.2f}度")
    if feature_diffs.get('max_x_factor_pro') is not None:
        print(f"   最大Xファクター（プロ）: {feature_diffs['max_x_factor_pro']:.2f}度")
    if feature_diffs.get('max_x_factor_diff') is not None:
        print(f"   最大Xファクターの差: {feature_diffs['max_x_factor_diff']:.2f}度")
    
    # 結果をJSONファイルに保存
    output_path = project_root / "output" / "comparison_result.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # JSONにシリアライズ可能な形式に変換（numpy型をPython標準型に変換）
    def convert_to_json_serializable(obj):
        """numpy型をPython標準型に変換"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif obj is None:
            return None
        else:
            return obj
    
    json_result = {
        'my_swing': result['my_swing'],
        'pro_swing': result['pro_swing'],
        'comparison': convert_to_json_serializable(result['comparison'])
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n6. 比較結果を保存しました: {output_path}")
    print("\n完了しました！")


if __name__ == "__main__":
    test_comparison()

