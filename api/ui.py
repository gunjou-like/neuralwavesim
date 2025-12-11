import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="Neural Wave Simulator",
    page_icon="🌊",
    layout="wide"
)

# API エンドポイント
API_URL = "http://127.0.0.1:8000"

# ========================================
# ユーティリティ関数
# ========================================

def calculate_energy_components(wave_history, dt, dx, c):
    """
    運動エネルギーとポテンシャルエネルギーを分離計算
    
    Args:
        wave_history: (nt, nx) の波形データ
        dt: 時間刻み
        dx: 空間刻み
        c: 波の速度
    
    Returns:
        kinetic: 運動エネルギー (nt-2,)
        potential: ポテンシャルエネルギー (nt-2,)
        total: 総エネルギー (nt-2,)
    """
    nt, nx = wave_history.shape
    
    kinetic = np.zeros(nt - 2)
    potential = np.zeros(nt - 2)
    total = np.zeros(nt - 2)
    
    for t in range(1, nt - 1):
        # 運動エネルギー: 0.5 * ∫ (∂u/∂t)^2 dx
        # 中心差分で時間微分を計算
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
        K = 0.5 * np.sum(u_t**2) * dx
        
        # ポテンシャルエネルギー: 0.5 * c^2 * ∫ (∂u/∂x)^2 dx
        # numpy の gradient で空間微分を計算
        u_x = np.gradient(wave_history[t], dx)
        P = 0.5 * c**2 * np.sum(u_x**2) * dx
        
        kinetic[t-1] = K
        potential[t-1] = P
        total[t-1] = K + P
    
    return kinetic, potential, total

# ========================================
# メイン UI
# ========================================

st.title("🌊 Neural Wave Simulator")
st.markdown("**3つのモデルで波動シミュレーションを比較**")

# サイドバー: パラメータ設定
st.sidebar.header("⚙️ シミュレーション設定")

model_type = st.sidebar.selectbox(
    "モデル選択",
    ["physics", "data-driven", "pinns"],
    format_func=lambda x: {
        "physics": "🔬 物理ベース (差分法)",
        "data-driven": "🧠 データ駆動型 (NN)",
        "pinns": "⚡ PINNs (物理制約付きNN)"
    }[x]
)

st.sidebar.subheader("物理パラメータ")
nx = st.sidebar.slider("空間グリッド数 (nx)", 50, 200, 100)
nt = st.sidebar.slider("時間ステップ数 (nt)", 50, 500, 200)
c = st.sidebar.slider("波の速度 (c)", 0.5, 2.0, 1.0, 0.1)

st.sidebar.subheader("初期波形")
wave_type = st.sidebar.selectbox(
    "波形タイプ",
    ["gaussian", "custom"],
    format_func=lambda x: {"gaussian": "ガウスパルス", "custom": "カスタム"}[x]
)

if wave_type == "gaussian":
    center = st.sidebar.slider("中心位置", 0.0, 10.0, 5.0, 0.5)
    width = st.sidebar.slider("幅", 0.1, 3.0, 1.0, 0.1)
    height = st.sidebar.slider("高さ", 0.1, 3.0, 1.0, 0.1)
    custom_data = None
else:
    custom_data = st.sidebar.text_area(
        "波形データ (カンマ区切り)",
        "0.5,1.0,0.5,0,0,..."
    )
    center, width, height = 5.0, 1.0, 1.0

# シミュレーション実行
if st.sidebar.button("🚀 シミュレーション実行", type="primary"):
    with st.spinner(f"{model_type} モデルで計算中..."):
        # リクエストペイロード
        payload = {
            "model_type": model_type,
            "nx": nx,
            "nt": nt,
            "c": c,
            "initial_condition": {
                "wave_type": wave_type,
                "center": center,
                "width": width,
                "height": height
            }
        }
        
        if wave_type == "custom" and custom_data:
            payload["initial_condition"]["data"] = [
                float(x.strip()) for x in custom_data.split(",") if x.strip()
            ]
        
        try:
            # API リクエスト
            response = requests.post(f"{API_URL}/simulate", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 結果を session_state に保存
            st.session_state.result = result
            st.success(f"✅ 計算完了！ ({result['computation_time_ms']:.2f} ms)")
        
        except requests.exceptions.HTTPError as e:
            # ★ API からのバリデーションエラーを表示
            if response.status_code == 422:
                try:
                    error_detail = response.json()
                    st.error("❌ パラメータエラー")
                    
                    # FastAPI の ValidationError を解析
                    if "detail" in error_detail:
                        for error in error_detail["detail"]:
                            loc = " → ".join(str(x) for x in error.get("loc", []))
                            msg = error.get("msg", "")
                            st.warning(f"**{loc}**: {msg}")
                except:
                    st.error(f"❌ バリデーションエラー: {e}")
            
            elif response.status_code == 400:
                # ★ 初期条件の検証エラー
                try:
                    error_detail = response.json()
                    st.error("❌ 初期条件が不適切です")
                    
                    if "detail" in error_detail:
                        error_msg = error_detail["detail"]
                        
                        # エラーメッセージを解析して具体的なアドバイス
                        if "境界に近すぎます" in error_msg:
                            st.warning("🔧 **修正案:**")
                            st.info(f"""
                            - 現在の中心位置: {center:.2f}
                            - 現在のパルス幅: {width:.2f}
                            - 推奨範囲: {max(3*width, 0.5):.2f} ≤ center ≤ {10.0 - max(3*width, 0.5):.2f}
                            
                            **対処法:**
                            1. 中心位置を領域の中央寄り（5.0付近）に設定
                            2. パルス幅を小さくする
                            """)
                        
                        elif "狭すぎます" in error_msg:
                            min_width = 10 * 0.1 / (2 * 3.14159)  # 概算
                            st.warning("🔧 **修正案:**")
                            st.info(f"""
                            - 現在のパルス幅: {width:.2f}
                            - 最小推奨幅: {min_width:.2f}
                            
                            **理由:**
                            - 空間解像度（dx={10.0/nx:.3f}）に対して幅が小さすぎます
                            - 数値分散により精度が低下します
                            
                            **対処法:**
                            1. パルス幅を {min_width*1.5:.2f} 以上に設定
                            2. または空間グリッド数を増やす（nx > {int(nx*1.5)}）
                            """)
                        
                        elif "広すぎます" in error_msg:
                            max_width = 10.0 / 4
                            st.warning("🔧 **修正案:**")
                            st.info(f"""
                            - 現在のパルス幅: {width:.2f}
                            - 最大推奨幅: {max_width:.2f}
                            
                            **理由:**
                            - 領域長（L=10.0）に対して幅が大きすぎます
                            - 境界反射波との干渉で予期しない共鳴が発生します
                            
                            **対処法:**
                            1. パルス幅を {max_width:.2f} 以下に設定
                            """)
                        
                        # 元のエラーメッセージも表示
                        with st.expander("詳細なエラーメッセージ"):
                            st.code(error_msg)
                except:
                    st.error(f"❌ リクエストエラー: {e}")
            
            else:
                st.error(f"❌ サーバーエラー: {e}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ API サーバーに接続できません")
            st.info("""
            **解決方法:**
            1. API サーバーが起動しているか確認してください
            
            ```bash
            # 別ターミナルで実行
            uvicorn api.main:app --reload
            ```
            
            2. ポート 8000 が使用可能か確認してください
            """)
        
        except requests.exceptions.Timeout:
            st.error("❌ リクエストがタイムアウトしました")
            st.info("計算時間が長すぎる可能性があります。nt を減らしてみてください。")
        
        except Exception as e:
            st.error(f"❌ 予期しないエラー: {e}")
            with st.expander("デバッグ情報"):
                st.json(payload)
                import traceback
                st.code(traceback.format_exc())

# 結果表示
if "result" in st.session_state:
    result = st.session_state.result
    wave_history = np.array(result["wave_history"])
    
    st.divider()
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("モデル", result["model_type"])
    col2.metric("データ形状", f"{wave_history.shape[0]} × {wave_history.shape[1]}")
    col3.metric("計算時間", f"{result['computation_time_ms']:.2f} ms")
    col4.metric("波の速度", f"{result['params']['c']}")
    
    # タブで表示切り替え
    tab1, tab2, tab3 = st.tabs(["📊 時系列可視化", "🎬 アニメーション", "📈 統計情報"])
    
    with tab1:
        st.subheader("時系列プロット")
        
        # Plotly による対話的プロット
        fig = go.Figure()
        
        # 複数時刻をプロット
        time_indices = np.linspace(0, nt-1, 6, dtype=int)
        x_grid = np.linspace(0, result['params']['L'], nx)
        
        for t_idx in time_indices:
            fig.add_trace(go.Scatter(
                x=x_grid,
                y=wave_history[t_idx],
                mode='lines',
                name=f't = {t_idx}'
            ))
        
        fig.update_layout(
            title="波形の時間変化",
            xaxis_title="位置 (x)",
            yaxis_title="変位 (u)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("時空間ヒートマップ")
        
        # ヒートマップ
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=wave_history,
            x=np.linspace(0, result['params']['L'], nx),
            y=np.linspace(0, result['params']['T_max'], nt),
            colorscale='RdBu',
            zmid=0
        ))
        
        fig_heatmap.update_layout(
            title="時空間分布",
            xaxis_title="位置 (x)",
            yaxis_title="時刻 (t)",
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.subheader("📊 統計情報とエネルギー解析")
        
        try:
            # エネルギー計算
            K, P, E = calculate_energy_components(
                wave_history,
                dt=result['params']['dt'],
                dx=result['params']['dx'],
                c=result['params']['c']
            )
            
            # 配列長チェック
            if len(E) == 0:
                st.warning("⚠️ エネルギー計算には最低3ステップ必要です")
            else:
                # メトリクス表示
                col1, col2, col3 = st.columns(3)
                
                E_var = (np.max(E) - np.min(E)) / np.mean(E) * 100
                
                with col1:
                    st.metric("平均総エネルギー", f"{np.mean(E):.4f}")
                    
                    if E_var < 1.0:
                        st.metric("エネルギー変動率", f"{E_var:.2f}%", delta="優秀", delta_color="normal")
                    elif E_var < 5.0:
                        st.metric("エネルギー変動率", f"{E_var:.2f}%", delta="許容範囲", delta_color="off")
                    else:
                        st.metric("エネルギー変動率", f"{E_var:.2f}%", delta="要改善", delta_color="inverse")
                
                with col2:
                    st.metric("運動エネルギー (K)", f"{np.mean(K):.4f}")
                    st.metric("K/E 比率", f"{np.mean(K)/np.mean(E)*100:.1f}%")
                
                with col3:
                    st.metric("ポテンシャル (P)", f"{np.mean(P):.4f}")
                    st.metric("P/E 比率", f"{np.mean(P)/np.mean(E)*100:.1f}%")
                
                # エネルギー時系列プロット
                fig_energy = go.Figure()
                
                time_axis = np.arange(len(E)) * result['params']['dt']
                
                fig_energy.add_trace(go.Scatter(
                    x=time_axis, y=K,
                    mode='lines',
                    name='運動エネルギー (K)',
                    line=dict(color='blue', width=2)
                ))
                
                fig_energy.add_trace(go.Scatter(
                    x=time_axis, y=P,
                    mode='lines',
                    name='ポテンシャルエネルギー (P)',
                    line=dict(color='red', width=2)
                ))
                
                fig_energy.add_trace(go.Scatter(
                    x=time_axis, y=E,
                    mode='lines',
                    name='総エネルギー (E = K + P)',
                    line=dict(color='black', width=3)
                ))
                
                fig_energy.add_hline(
                    y=np.mean(E),
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"平均値 (保存則)"
                )
                
                fig_energy.update_layout(
                    title="エネルギー成分の時間変化",
                    xaxis_title="時刻 (t)",
                    yaxis_title="エネルギー",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_energy, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ エネルギー計算エラー: {e}")
            st.info("デバッグ情報:")
            st.json({
                "wave_history.shape": wave_history.shape,
                "dt": result['params']['dt'],
                "dx": result['params']['dx'],
                "c": result['params']['c']
            })