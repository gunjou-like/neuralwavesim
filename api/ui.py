import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ✅ API URL を環境変数から取得（Docker 用）
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="NeuralWaveSim", layout="wide")

# Title
st.title("🌊 NeuralWaveSim - 波動シミュレーション")
st.markdown("物理ベースとニューラルネットワークによる波動方程式のシミュレーション")

# ✅ Debug info in sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("🔧 接続状態", expanded=True):
        st.text(f"API URL: {API_URL}")
        
        # API 接続テスト
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API 接続成功")
                health_data = health_response.json()
                st.json(health_data)
            else:
                st.error(f"❌ API 接続失敗")
                st.text(f"Status Code: {health_response.status_code}")
        except requests.exceptions.ConnectionError as e:
            st.error("❌ API に接続できません")
            st.code(str(e))
            st.info("docker-compose logs api でログを確認してください")
        except requests.exceptions.Timeout:
            st.error("❌ 接続タイムアウト")
        except Exception as e:
            st.error(f"❌ エラー: {e}")

def run_simulation(config: dict):
    """Run simulation via API"""
    try:
        response = requests.post(
            f"{API_URL}/simulate",
            json=config,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP エラー: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            st.error(f"レスポンス: {http_err.response.text}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"❌ APIサーバーに接続できません")
        st.info(f"API URL: {API_URL}")
        st.code(str(conn_err))
        st.info("docker-compose logs api でログを確認してください")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ リクエストがタイムアウトしました")
        return None
    except Exception as e:
        st.error(f"❌ エラー: {str(e)}")
        return None

# Model selection
model_type = st.sidebar.selectbox(
    "モデル選択",
    ["physics", "data-driven", "pinns", "pinns-v2"],
    index=0
)

# Model descriptions
model_info = {
    "physics": "物理ベース（有限差分法）",
    "data-driven": "データ駆動型ニューラルネットワーク",
    "pinns": "物理法則組込みNN（オリジナル）",
    "pinns-v2": "物理法則組込みNN v2（エネルギー保存改善版）⭐"
}
st.sidebar.info(f"**{model_info[model_type]}**")

# Physics parameters
nx = st.sidebar.slider("空間グリッド数 (nx)", 50, 200, 100)
nt = st.sidebar.slider("時間ステップ数 (nt)", 100, 400, 200)
c = st.sidebar.slider("波速 (c)", 0.5, 2.0, 1.0, 0.1)

# Initial condition
st.sidebar.subheader("初期条件")
wave_type = st.sidebar.selectbox("波形タイプ", ["gaussian", "sine", "custom"])

center = st.sidebar.slider("中心位置", 0.0, 10.0, 5.0, 0.1)
width = st.sidebar.slider("幅", 0.1, 3.0, 1.0, 0.1)
height = st.sidebar.slider("高さ", 0.1, 2.0, 1.0, 0.1)

# Run simulation
if st.sidebar.button("シミュレーション実行", type="primary"):
    with st.spinner("計算中..."):
        # ✅ run_simulation() 関数を使用
        config = {
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
        
        result = run_simulation(config)
        
        if result:
            wave_history = np.array(result["wave_history"])
            params = result["params"]
            comp_time = result["computation_time_ms"]
            
            st.success(f"✅ シミュレーション完了 ({comp_time:.2f} ms)")
            
            # Store results
            st.session_state.wave_history = wave_history
            st.session_state.params = params
            st.session_state.model_type = model_type
            st.session_state.comp_time = comp_time
        else:
            st.error("シミュレーションに失敗しました")

# Display results
if "wave_history" in st.session_state:
    wave_history = st.session_state.wave_history
    params = st.session_state.params
    model_type = st.session_state.model_type
    comp_time = st.session_state.comp_time
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("モデル", model_type)
    col2.metric("計算時間", f"{comp_time:.2f} ms")
    col3.metric("グリッドサイズ", f"{params['nx']} x {params['nt']}")
    
    # 3 Tabs: 波形、アニメーション、統計データ
    tab1, tab2, tab3 = st.tabs(["📈 波形", "🎬 アニメーション", "📊 統計データ"])
    
    with tab1:
        # Heatmap
        st.subheader("時空間発展")
        
        t = np.arange(params['nt']) * params['dt']
        x = np.linspace(0, params['L'], params['nx'])
        
        fig = go.Figure(data=go.Heatmap(
            z=wave_history,
            x=x,
            y=t,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="変位 u")
        ))
        
        fig.update_layout(
            title="波の時空間発展",
            xaxis_title="位置 x (m)",
            yaxis_title="時刻 t (s)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Animation with time slider
        st.subheader("波形アニメーション")
        
        time_idx = st.slider(
            "時刻を選択",
            0,
            params['nt'] - 1,
            0,
            key="time_slider"
        )
        
        current_time = time_idx * params['dt']
        x = np.linspace(0, params['L'], params['nx'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=wave_history[time_idx],
            mode='lines',
            name=f't = {current_time:.2f} s',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title=f"波形スナップショット (t = {current_time:.2f} s)",
            xaxis_title="位置 x (m)",
            yaxis_title="変位 u",
            yaxis_range=[-height * 1.2, height * 1.2],
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show grid info
        st.caption(f"時間ステップ: {time_idx + 1} / {params['nt']}")
    
    with tab3:
        # Statistics
        st.subheader("統計データ")
        
        # Energy analysis
        st.markdown("#### エネルギー保存解析")
        
        energies = []
        kinetic_energies = []
        potential_energies = []
        
        for i in range(1, params['nt'] - 1):
            u_t = (wave_history[i+1] - wave_history[i-1]) / (2 * params['dt'])
            u_x = np.gradient(wave_history[i], params['dx'])
            
            K = 0.5 * np.sum(u_t**2) * params['dx']
            P = 0.5 * params['c']**2 * np.sum(u_x**2) * params['dx']
            E = K + P
            
            kinetic_energies.append(K)
            potential_energies.append(P)
            energies.append(E)
        
        energies = np.array(energies)
        kinetic_energies = np.array(kinetic_energies)
        potential_energies = np.array(potential_energies)
        
        E_mean = np.mean(energies)
        E_std = np.std(energies)
        E_variation = (np.max(energies) - np.min(energies)) / E_mean * 100
        
        # Energy metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("平均エネルギー", f"{E_mean:.4f}")
        col2.metric("標準偏差", f"{E_std:.6f}")
        col3.metric("変動率", f"{E_variation:.2f}%")
        
        if E_variation < 5.0:
            col4.metric("評価", "✅ 優秀")
        elif E_variation < 10.0:
            col4.metric("評価", "⚠️ 許容範囲")
        else:
            col4.metric("評価", "❌ 要改善")
        
        # Energy plot
        time_points = np.arange(1, params['nt'] - 1) * params['dt']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=energies,
            mode='lines',
            name='総エネルギー',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=kinetic_energies,
            mode='lines',
            name='運動エネルギー',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=potential_energies,
            mode='lines',
            name='位置エネルギー',
            line=dict(color='blue', width=1, dash='dash')
        ))
        
        fig.add_hline(
            y=E_mean,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"平均: {E_mean:.4f}"
        )
        
        fig.update_layout(
            title="エネルギー時間発展",
            xaxis_title="時刻 (s)",
            yaxis_title="エネルギー",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Wave statistics
        st.markdown("#### 波形統計")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("最大振幅", f"{np.max(np.abs(wave_history)):.4f}")
            st.metric("最小値", f"{np.min(wave_history):.4f}")
            st.metric("最大値", f"{np.max(wave_history):.4f}")
        
        with col2:
            st.metric("平均値", f"{np.mean(wave_history):.6f}")
            st.metric("標準偏差", f"{np.std(wave_history):.4f}")
            st.metric("データ点数", f"{wave_history.size}")