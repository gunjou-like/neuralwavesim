# app/ui.py
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
NX = 100
# ローカル開発用のAPI URL (後でクラウドのURLに書き換えます)
API_URL = "http://127.0.0.1:8000/predict"

st.title("🌊 1D Wave Equation AI Simulator")
st.caption("AIが物理法則（波動方程式）を再現します")

# --- 初期化 (Session State) ---
# 画面がリロードされても変数を保持するための仕組み
if 'wave_curr' not in st.session_state:
    # 初期状態: 真ん中にガウス波形
    x = np.linspace(0, 10, NX)
    st.session_state['wave_curr'] = np.exp(-(x - 5)**2 / 0.5)
    st.session_state['wave_prev'] = st.session_state['wave_curr'].copy() # 初期速度0

# --- 画面描画 ---
col1, col2 = st.columns([3, 1])

with col1:
    # グラフ描画 (Matplotlib)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Waveform")
    ax.plot(st.session_state['wave_curr'], color='blue', label='AI Prediction')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.write("Controls")
    
    # リセットボタン
    if st.button("Reset Wave"):
        x = np.linspace(0, 10, NX)
        st.session_state['wave_curr'] = np.exp(-(x - 5)**2 / 0.5)
        st.session_state['wave_prev'] = st.session_state['wave_curr'].copy()
        st.rerun()

    # 進めるボタン (ここが重要！)
    if st.button("Step Forward (AI Predict)"):
        # 1. 入力データの作成 (現在 + 過去 を連結)
        input_data = np.concatenate([
            st.session_state['wave_curr'], 
            st.session_state['wave_prev']
        ]).tolist()
        
        # 2. APIに送信
        try:
            response = requests.post(API_URL, json={"wave_data": input_data})
            
            if response.status_code == 200:
                result = response.json()
                next_wave = np.array(result["next_wave"])
                
                # 3. 状態更新 (時間を進める)
                st.session_state['wave_prev'] = st.session_state['wave_curr']
                st.session_state['wave_curr'] = next_wave
                
                # 画面を更新
                st.rerun()
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("APIサーバーに接続できません。backendが起動しているか確認してください。")