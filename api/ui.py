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
st.title("🌊 NeuralWaveSim")
st.markdown("""
            **Physics-Informed Neural Networks for Wave Equation Simulation**
            This application demonstrates four approachs to solving the 1D wave equation:
            1. **Physics-Based Solver**: Traditional finite difference method.
            2. **Data-Driven Neural Network**: A neural network trained on pure simulation data.
            3. **PINNs (Original)**: Physics-Informed Neural Networks incorporating wave equation constraints.
            4. **PINNs v2**: An improved version of PINNs with enhanced energy conservation.
            """)

# ✅ Debug info in sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("🔧 Connection Status", expanded=True):
        st.text(f"API URL: {API_URL}")
        
        # API connection test
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API connection successful")
                health_data = health_response.json()
                st.json(health_data)
            else:
                st.error(f"❌ API connection failed")
                st.text(f"Status Code: {health_response.status_code}")
        except requests.exceptions.ConnectionError as e:
            st.error("❌ Cannot connect to API")
            st.code(str(e))
            st.info("Check logs with: docker-compose logs api")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out")
        except Exception as e:
            st.error(f"❌ Error: {e}")
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
        st.error(f"❌ HTTP error: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            st.error(f"Response: {http_err.response.text}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"❌ Cannot connect to API server")
        st.info(f"API URL: {API_URL}")
        st.code(str(conn_err))
        st.info("Check logs with: docker-compose logs api")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# Model selection
model_type = st.sidebar.selectbox(
    "Model Selection",
    ["physics", "data-driven-v2", "pinns", "pinns-v2"],
    index=0
)

# Model descriptions
model_info = {
    "physics": "Physics-Based (Finite Difference Method)",
    "data-driven-v2": "Data-Driven Neural Network v2 (LSTM-based)",
    "pinns": "Physics-Informed Neural Networks (Original)",
    "pinns-v2": "Physics-Informed Neural Networks v2 (Improved Energy Conservation) ⭐"
}
st.sidebar.info(f"**{model_info[model_type]}**")

# Physics parameters
nx = st.sidebar.slider("Spatial Grid Points (nx)", 50, 200, 100)
nt = st.sidebar.slider("Time Steps (nt)", 100, 400, 200)
c = st.sidebar.slider("Wave Speed (c)", 0.5, 2.0, 1.0, 0.1)

# Initial condition
st.sidebar.subheader("Initial Condition")
wave_type = st.sidebar.selectbox("Wave Type", ["gaussian", "sine", "custom"])
center = st.sidebar.slider("Center", 0.0, 10.0, 5.0, 0.1)
width = st.sidebar.slider("Width", 0.1, 3.0, 1.0, 0.1)
height = st.sidebar.slider("Height", 0.1, 2.0, 1.0, 0.1)

# Run simulation
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Computing..."):
        # ✅ run_simulation() function usage
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
            
            st.success(f"✅ Simulation Complete ({comp_time:.2f} ms)")
            
            # Store results
            st.session_state.wave_history = wave_history
            st.session_state.params = params
            st.session_state.model_type = model_type
            st.session_state.comp_time = comp_time
        else:
            st.error("❌ Simulation failed")

# Display results
if "wave_history" in st.session_state:
    wave_history = st.session_state.wave_history
    params = st.session_state.params
    model_type = st.session_state.model_type
    comp_time = st.session_state.comp_time
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_type)
    col2.metric("Computation Time", f"{comp_time:.2f} ms")
    col3.metric("Grid Size", f"{params['nx']} x {params['nt']}")
    
    # 3 Tabs: Waveform, Animation, Statistics
    tab1, tab2, tab3 = st.tabs(["📈 Waveform", "🎬 Animation", "📊 Statistics"])
    
    with tab1:
        # Heatmap
        st.subheader("Spatiotemporal Evolution")
        
        t = np.arange(params['nt']) * params['dt']
        x = np.linspace(0, params['L'], params['nx'])
        
        fig = go.Figure(data=go.Heatmap(
            z=wave_history,
            x=x,
            y=t,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Displacement u")
        ))
        
        fig.update_layout(
            title="Spatiotemporal Evolution of the Wave",
            xaxis_title="Position x (m)",
            yaxis_title="Time t (s)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Animation with time slider
        st.subheader("Waveform Animation")
        
        time_idx = st.slider(
            "Select Time",
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
            title=f"Waveform Snapshot (t = {current_time:.2f} s)",
            xaxis_title="Position x (m)",
            yaxis_title="Displacement u",
            yaxis_range=[-height * 1.2, height * 1.2],
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show grid info
        st.caption(f"Time Step: {time_idx + 1} / {params['nt']}")
    
    with tab3:
        # Statistics
        st.subheader("Statistics")
        
        # Energy analysis
        st.markdown("#### Energy Conservation Analysis")
        
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
        col1.metric("Average Energy", f"{E_mean:.4f}")
        col2.metric("Standard Deviation", f"{E_std:.6f}")
        col3.metric("Variation", f"{E_variation:.2f}%")
        
        if E_variation < 5.0:
            col4.metric("Evaluation", "✅ Excellent")
        elif E_variation < 10.0:
            col4.metric("Evaluation", "⚠️ Acceptable")
        else:
            col4.metric("Evaluation", "❌ Needs Improvement")
        
        # Energy plot
        time_points = np.arange(1, params['nt'] - 1) * params['dt']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=energies,
            mode='lines',
            name='Total Energy',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=kinetic_energies,
            mode='lines',
            name='Kinetic Energy',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=potential_energies,
            mode='lines',
            name='Potential Energy',
            line=dict(color='blue', width=1, dash='dash')
        ))
        
        fig.add_hline(
            y=E_mean,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Average: {E_mean:.4f}"
        )
        
        fig.update_layout(
            title="Energy Time Evolution",
            xaxis_title="Time (s)",
            yaxis_title="Energy",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Wave statistics
        st.markdown("#### Wave Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Maximum Amplitude", f"{np.max(np.abs(wave_history)):.4f}")
            st.metric("Minimum Value", f"{np.min(wave_history):.4f}")
            st.metric("Maximum Value", f"{np.max(wave_history):.4f}")
        
        with col2:
            st.metric("Mean Value", f"{np.mean(wave_history):.6f}")
            st.metric("Standard Deviation", f"{np.std(wave_history):.4f}")
            st.metric("Data Points", f"{wave_history.size}")