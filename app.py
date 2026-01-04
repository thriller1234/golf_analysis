"""
ゴルフスイング解析アプリ - Streamlit UI
"""

import streamlit as st
import sys
from pathlib import Path
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pose_estimation import PoseEstimator
from src.swing_analysis import PhaseDetector, SwingAnalyzer, SwingNormalizer
from src.visualization import SwingVisualizer
from src.club_detection import ClubDetector


# ページ設定
st.set_page_config(
    page_title="ゴルフスイング解析アプリ",
    page_icon="⛳",
    layout="wide"
)

# タイトル
st.title("⛳ ゴルフスイング解析アプリ")
st.markdown("後方から撮影した動画を使用して、スイングを解析・比較します。")

# サイドバー
st.sidebar.header("設定")

# タブ
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 動画解析",
    "📊 スイング比較",
    "📈 軌道可視化",
    "ℹ️ 使い方"
])

# タブ1: 動画解析
with tab1:
    st.header("動画解析")
    
    uploaded_file = st.file_uploader(
        "スイング動画をアップロードしてください",
        type=['mp4', 'avi', 'mov'],
        help="後方から撮影した動画を推奨します"
    )
    
    if uploaded_file is not None:
        # 動画を一時保存
        temp_path = project_root / "data" / "videos" / "temp_upload.mp4"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"動画をアップロードしました: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 姿勢推定を実行", type="primary"):
                with st.spinner("姿勢推定を実行中..."):
                    try:
                        estimator = PoseEstimator()
                        keypoints = estimator.process_video(temp_path)
                        
                        st.success(f"✅ 骨格情報を {len(keypoints)} フレーム分取得しました")
                        
                        # セッション状態に保存（簡易版：フレーム数と主要関節のみ）
                        st.session_state['keypoints_count'] = len(keypoints)
                        st.session_state['video_path'] = str(temp_path)
                        st.session_state['keypoints_full'] = keypoints  # フルデータも保存
                        
                        # 最初のフレームの主要関節を表示
                        if keypoints:
                            first_frame = keypoints[0]
                            estimator_instance = estimator
                            key_joints = estimator_instance.get_key_joints(
                                first_frame['landmarks']
                            )
                            
                            st.subheader("主要関節の座標（最初のフレーム）")
                            joint_df = {
                                '関節': list(key_joints.keys()),
                                'X座標': [pos[0] for pos in key_joints.values()],
                                'Y座標': [pos[1] for pos in key_joints.values()]
                            }
                            st.dataframe(joint_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
        
        with col2:
            if st.button("🎯 スイング位相を検出"):
                if 'keypoints_full' not in st.session_state:
                    st.warning("先に姿勢推定を実行してください")
                else:
                    with st.spinner("スイング位相を検出中..."):
                        try:
                            phase_detector = PhaseDetector()
                            phases = phase_detector.detect_phases(st.session_state['keypoints_full'])
                            
                            st.success("✅ スイング位相を検出しました")
                            
                            # 位相を表示
                            phase_df = {
                                '位相': ['アドレス', 'トップ', 'インパクト', 'フィニッシュ'],
                                'フレーム番号': [
                                    phases['address'],
                                    phases['top'],
                                    phases['impact'],
                                    phases['finish']
                                ]
                            }
                            st.dataframe(phase_df, use_container_width=True)
                            
                            # セッション状態に保存
                            st.session_state['phases'] = phases
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

# タブ2: スイング比較
with tab2:
    st.header("スイング比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("自分のスイング")
        my_video = st.file_uploader(
            "自分のスイング動画",
            type=['mp4', 'avi', 'mov'],
            key="my_swing"
        )
    
    with col2:
        st.subheader("プロ選手のスイング")
        pro_video = st.file_uploader(
            "プロ選手のスイング動画",
            type=['mp4', 'avi', 'mov'],
            key="pro_swing"
        )
    
    if my_video and pro_video:
        if st.button("🔄 比較解析を実行", type="primary"):
            with st.spinner("比較解析を実行中..."):
                try:
                    # 動画を保存
                    my_path = project_root / "data" / "videos" / "temp_my.mp4"
                    pro_path = project_root / "data" / "videos" / "temp_pro.mp4"
                    
                    with open(my_path, "wb") as f:
                        f.write(my_video.getbuffer())
                    with open(pro_path, "wb") as f:
                        f.write(pro_video.getbuffer())
                    
                    # 比較解析
                    analyzer = SwingAnalyzer()
                    result = analyzer.compare_swings(my_path, pro_path)
                    
                    st.success("✅ 比較解析が完了しました")
                    
                    # 結果を表示
                    st.subheader("比較結果")
                    
                    # 位相の比較
                    my_phases = result['comparison']['my_phases']
                    pro_phases = result['comparison']['pro_phases']
                    
                    phase_comparison_df = {
                        '位相': ['アドレス', 'トップ', 'インパクト', 'フィニッシュ'],
                        '自分のフレーム': [
                            my_phases['address'],
                            my_phases['top'],
                            my_phases['impact'],
                            my_phases['finish']
                        ],
                        'プロのフレーム': [
                            pro_phases['address'],
                            pro_phases['top'],
                            pro_phases['impact'],
                            pro_phases['finish']
                        ]
                    }
                    st.dataframe(phase_comparison_df, use_container_width=True)
                    
                    # 特徴量の比較
                    st.subheader("特徴量の比較")
                    feature_diffs = result['comparison']['feature_differences']
                    
                    if feature_diffs.get('max_x_factor_my') is not None:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("最大Xファクター（自分）", 
                                    f"{feature_diffs['max_x_factor_my']:.2f}°")
                        with col_b:
                            st.metric("最大Xファクター（プロ）", 
                                    f"{feature_diffs['max_x_factor_pro']:.2f}°")
                        with col_c:
                            diff = feature_diffs.get('max_x_factor_diff', 0)
                            st.metric("差分", f"{diff:.2f}°", 
                                    delta=f"{abs(diff):.2f}°" if diff != 0 else None)
                    
                    # 可視化
                    st.subheader("比較グラフ")
                    visualizer = SwingVisualizer()
                    comparison_path = project_root / "output" / "streamlit_comparison.png"
                    comparison_path.parent.mkdir(exist_ok=True)
                    
                    # キーポイントを取得（比較用）
                    estimator = PoseEstimator()
                    my_kp = estimator.process_video(my_path)
                    pro_kp = estimator.process_video(pro_path)
                    
                    visualizer.compare_swings_side_by_side(
                        my_kp,
                        pro_kp,
                        my_phases=my_phases,
                        pro_phases=pro_phases,
                        save_path=comparison_path
                    )
                    
                    st.image(str(comparison_path), width='stretch')
                    
                    # 結果をJSONでダウンロード可能にする
                    json_result = {
                        'my_swing': str(my_path),
                        'pro_swing': str(pro_path),
                        'comparison': result['comparison']
                    }
                    
                    # JSONにシリアライズ可能な形式に変換
                    def convert_to_json_serializable(obj):
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
                    
                    json_result = convert_to_json_serializable(json_result)
                    json_str = json.dumps(json_result, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        label="📥 比較結果をJSONでダウンロード",
                        data=json_str,
                        file_name="comparison_result.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    st.exception(e)

# タブ3: 軌道可視化
with tab3:
    st.header("スイング軌道可視化")
    
    if 'keypoints_full' in st.session_state and 'phases' in st.session_state:
        if st.button("📊 軌道グラフを生成"):
            with st.spinner("軌道グラフを生成中..."):
                try:
                    visualizer = SwingVisualizer()
                    trajectory_path = project_root / "output" / "streamlit_trajectory.png"
                    
                    visualizer.plot_swing_trajectory(
                        st.session_state['keypoints_full'],
                        phases=st.session_state['phases'],
                        save_path=trajectory_path
                    )
                    
                    st.success("✅ 軌道グラフを生成しました")
                    st.image(str(trajectory_path), width='stretch')
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    else:
        st.info("先に「動画解析」タブで姿勢推定と位相検出を実行してください")

# タブ4: 使い方
with tab4:
    st.header("使い方")
    
    st.markdown("""
    ## 基本的な使い方
    
    ### 1. 動画解析
    1. 「動画解析」タブで動画をアップロード
    2. 「姿勢推定を実行」ボタンをクリック
    3. 「スイング位相を検出」ボタンをクリック
    
    ### 2. スイング比較
    1. 「スイング比較」タブで自分のスイングとプロ選手のスイングをアップロード
    2. 「比較解析を実行」ボタンをクリック
    3. 結果を確認
    
    ### 3. 軌道可視化
    1. 「軌道可視化」タブで軌道グラフを生成
    2. スイングの軌道を視覚的に確認
    
    ## 動画の要件
    
    - **フレームレート**: 120fps以上を推奨（最低60fps）
    - **解像度**: 1080p以上を推奨
    - **撮影角度**: 後方（Down-the-Line）から撮影
    
    ## 注意事項
    
    - インパクト検出は近似値です（クラブ検出により精度向上可能）
    - 解析には時間がかかる場合があります
    - 大きな動画ファイルは処理に時間がかかります
    """)

# フッター
st.markdown("---")
st.markdown("**ゴルフスイング解析アプリ** | 詳細は `docs/howtodo.md` を参照してください")

