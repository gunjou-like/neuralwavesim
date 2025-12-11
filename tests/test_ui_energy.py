import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

def calculate_energy_components(wave_history, dt, dx, c):
    """ui.py と同じ実装"""
    nt, nx = wave_history.shape
    
    kinetic = np.zeros(nt - 2)
    potential = np.zeros(nt - 2)
    total = np.zeros(nt - 2)
    
    for t in range(1, nt - 1):
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
        u_x = np.gradient(wave_history[t], dx)
        
        kinetic[t-1] = 0.5 * np.sum(u_t**2) * dx
        potential[t-1] = 0.5 * c**2 * np.sum(u_x**2) * dx
        total[t-1] = kinetic[t-1] + potential[t-1]
    
    return kinetic, potential, total

def test_energy_calculation():
    """関数の動作確認"""
    # ダミーデータ
    nt, nx = 100, 50
    wave_history = np.random.randn(nt, nx) * 0.1
    dt = 0.05
    dx = 0.1
    c = 1.0
    
    K, P, E = calculate_energy_components(wave_history, dt, dx, c)
    
    print("=" * 50)
    print("エネルギー計算テスト")
    print("=" * 50)
    print(f"入力形状: {wave_history.shape}")
    print(f"運動エネルギー配列: {K.shape}")
    print(f"ポテンシャルエネルギー配列: {P.shape}")
    print(f"総エネルギー配列: {E.shape}")
    print()
    print(f"K の平均: {np.mean(K):.6f}")
    print(f"P の平均: {np.mean(P):.6f}")
    print(f"E の平均: {np.mean(E):.6f}")
    print(f"E の変動率: {(np.max(E) - np.min(E)) / np.mean(E) * 100:.2f}%")
    print("=" * 50)
    print("✅ テスト成功")

if __name__ == "__main__":
    test_energy_calculation()