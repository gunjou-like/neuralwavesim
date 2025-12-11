import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition

def calculate_energy_components(wave_history, dt, dx, c):
    """
    運動エネルギーとポテンシャルエネルギーを分離計算
    
    Returns:
        kinetic: 運動エネルギー
        potential: ポテンシャルエネルギー
        total: 総エネルギー
    """
    nt, nx = wave_history.shape
    
    kinetic = np.zeros(nt - 2)
    potential = np.zeros(nt - 2)
    total = np.zeros(nt - 2)
    
    for t in range(1, nt - 1):
        # 運動エネルギー: 0.5 * ∫ (∂u/∂t)^2 dx
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
        K = 0.5 * np.sum(u_t**2) * dx
        
        # ポテンシャルエネルギー: 0.5 * c^2 * ∫ (∂u/∂x)^2 dx
        u_x = np.gradient(wave_history[t], dx)
        P = 0.5 * c**2 * np.sum(u_x**2) * dx
        
        kinetic[t-1] = K
        potential[t-1] = P
        total[t-1] = K + P
    
    return kinetic, potential, total

def analyze_frequency_spectrum(E, dt):
    """
    エネルギー変動の周波数スペクトルを解析
    """
    from scipy.fft import fft, fftfreq
    
    # FFT実行
    E_variation = E - np.mean(E)
    N = len(E_variation)
    
    # ゼロパディング（解像度向上）
    N_padded = 2**int(np.ceil(np.log2(N)) + 2)
    E_padded = np.pad(E_variation, (0, N_padded - N), mode='constant')
    
    yf = fft(E_padded)
    xf = fftfreq(N_padded, dt)[:N_padded//2]
    
    # パワースペクトル
    power = 2.0/N_padded * np.abs(yf[:N_padded//2])
    
    # ピーク検出
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, height=np.max(power)*0.1, distance=10)
    
    return xf, power, peaks

def test_energy_periodicity():
    """エネルギーの周期性を検証"""
    print("=" * 60)
    print("エネルギー成分の周期性検証")
    print("=" * 60)
    
    # パラメータ設定
    params = PhysicsParams(
        nx=100,
        nt=400,  # より長い時間シミュレーション
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=20.0
    )
    
    print(f"CFL数: {params.courant_number:.4f}")
    print(f"シミュレーション時間: {params.T_max} 秒")
    print()
    
    # 初期条件
    ic = InitialCondition(wave_type="gaussian", center=5.0, width=1.0, height=1.0)
    
    # シミュレーション実行
    solver = WaveSolver(params)
    wave_history = solver.solve(ic)
    
    # エネルギー計算
    K, P, E = calculate_energy_components(wave_history, params.dt, params.dx, params.c)
    
    # 時間軸
    time = np.arange(len(E)) * params.dt
    
    # 統計情報
    print("総エネルギー (E = K + P):")
    print(f"  平均値: {np.mean(E):.6f}")
    print(f"  標準偏差: {np.std(E):.6f} ({np.std(E)/np.mean(E)*100:.3f}%)")
    print(f"  変動範囲: {(np.max(E) - np.min(E))/np.mean(E)*100:.3f}%")
    print()
    
    print("運動エネルギー (K):")
    print(f"  平均値: {np.mean(K):.6f}")
    print(f"  最大値: {np.max(K):.6f}")
    print(f"  最小値: {np.min(K):.6f}")
    print()
    
    print("ポテンシャルエネルギー (P):")
    print(f"  平均値: {np.mean(P):.6f}")
    print(f"  最大値: {np.max(P):.6f}")
    print(f"  最小値: {np.min(P):.6f}")
    print()
    
    # 周期性の検出（FFT）
    from scipy.signal import find_peaks
    
    # 総エネルギーの変動成分
    E_variation = E - np.mean(E)
    
    # ピーク検出
    peaks, _ = find_peaks(E_variation, distance=10)
    if len(peaks) > 1:
        periods = np.diff(peaks) * params.dt
        avg_period = np.mean(periods)
        print(f"周期性の検出:")
        print(f"  平均周期: {avg_period:.3f} 秒")
        print(f"  理論値（定在波の周期）: {2*params.L/params.c:.3f} 秒")
        print()
    
    # 周波数解析を追加
    print("\n周波数解析:")
    print("=" * 60)
    
    frequencies, power, peaks = analyze_frequency_spectrum(E, params.dt)
    
    if len(peaks) > 0:
        # 支配的な周波数
        dominant_freq = frequencies[peaks[0]]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
        
        print(f"支配的な周波数: {dominant_freq:.4f} Hz")
        print(f"対応する周期: {dominant_period:.3f} 秒")
        print()
        
        # 理論周波数の計算
        print("理論的な固有周波数:")
        for n in range(1, 6):
            freq_n = n * params.c / (2 * params.L)
            period_n = 1.0 / freq_n
            print(f"  第{n}モード: f = {freq_n:.4f} Hz, T = {period_n:.3f} 秒")
        print()
        
        # ピークの詳細
        print(f"検出されたピーク数: {len(peaks)}")
        for i, peak_idx in enumerate(peaks[:5]):  # 上位5個
            freq = frequencies[peak_idx]
            period = 1.0 / freq if freq > 0 else float('inf')
            amplitude = power[peak_idx]
            print(f"  ピーク{i+1}: f = {freq:.4f} Hz, T = {period:.3f} 秒, 振幅 = {amplitude:.6f}")
    
    # 可視化
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # (1) 時空間分布
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(wave_history, aspect='auto', cmap='RdBu', 
                    extent=[0, params.L, 0, params.T_max], origin='lower')
    ax1.set_title('波動の時空間分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('位置 (x)')
    ax1.set_ylabel('時刻 (t)')
    plt.colorbar(im, ax=ax1, label='変位 (u)')
    
    # (2) エネルギー成分の時間変化
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, K, 'b-', label='運動エネルギー (K)', linewidth=1.5, alpha=0.8)
    ax2.plot(time, P, 'r-', label='ポテンシャルエネルギー (P)', linewidth=1.5, alpha=0.8)
    ax2.plot(time, E, 'k-', label='総エネルギー (E = K + P)', linewidth=2)
    ax2.axhline(np.mean(E), color='gray', linestyle='--', alpha=0.5, label='E の平均値')
    ax2.set_title('エネルギー成分の時間変化', fontsize=12, fontweight='bold')
    ax2.set_xlabel('時刻 (t)')
    ax2.set_ylabel('エネルギー')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # (3) 総エネルギーの詳細
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, E, 'k-', linewidth=2)
    ax3.axhline(np.mean(E), color='r', linestyle='--', linewidth=1.5, label='平均値')
    ax3.fill_between(time, np.mean(E) - np.std(E), np.mean(E) + np.std(E), 
                      alpha=0.2, color='red', label='±1σ')
    ax3.set_title(f'総エネルギーの変動 (変動率 {(np.max(E)-np.min(E))/np.mean(E)*100:.2f}%)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('時刻 (t)')
    ax3.set_ylabel('総エネルギー (E)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # (4) K と P の相関
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(K, P, c=time, cmap='viridis', s=10, alpha=0.6)
    ax4.plot([0, np.max(K)], [np.mean(E), np.mean(E) - np.max(K)], 
             'r--', linewidth=2, label='K + P = E (理論)')
    ax4.set_title('運動 vs ポテンシャルエネルギー', fontsize=12, fontweight='bold')
    ax4.set_xlabel('運動エネルギー (K)')
    ax4.set_ylabel('ポテンシャルエネルギー (P)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4, label='時刻 (t)')
    
    # (5) エネルギー比率
    ax5 = fig.add_subplot(gs[2, 1])
    K_ratio = K / E * 100
    P_ratio = P / E * 100
    ax5.plot(time, K_ratio, 'b-', label='K / E (%)', linewidth=1.5, alpha=0.8)
    ax5.plot(time, P_ratio, 'r-', label='P / E (%)', linewidth=1.5, alpha=0.8)
    ax5.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% (平均)')
    ax5.set_title('エネルギー比率の時間変化', fontsize=12, fontweight='bold')
    ax5.set_xlabel('時刻 (t)')
    ax5.set_ylabel('比率 (%)')
    ax5.set_ylim([0, 100])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # (6) 周波数スペクトル（新規追加）
    ax6 = fig.add_subplot(gs[3, :])
    ax6.semilogy(frequencies, power, 'b-', linewidth=1.5)
    
    # ピークをマーク
    if len(peaks) > 0:
        ax6.plot(frequencies[peaks], power[peaks], 'ro', markersize=8, label='検出されたピーク')
    
    # 理論値をプロット
    for n in range(1, 6):
        freq_theory = n * params.c / (2 * params.L)
        ax6.axvline(freq_theory, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax6.text(freq_theory, np.max(power)*0.5, f'n={n}', 
                rotation=90, va='bottom', fontsize=8)
    
    ax6.set_xlabel('周波数 (Hz)', fontsize=12)
    ax6.set_ylabel('パワースペクトル', fontsize=12)
    ax6.set_title('エネルギー変動の周波数解析', fontsize=12, fontweight='bold')
    ax6.set_xlim([0, 1.0])  # 0-1 Hz の範囲
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig('energy_periodicity_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 図を保存: energy_periodicity_analysis.png")
    
    # 結論
    print("=" * 60)
    print("結論:")
    print("=" * 60)
    
    if (np.max(E) - np.min(E)) / np.mean(E) < 0.05:
        print("✅ 総エネルギーはほぼ保存されている")
        print("   周期的変動は K と P の正常な交換によるもの")
        print()
        print("物理的解釈:")
        print("  - 運動エネルギーとポテンシャルエネルギーが周期的に変換")
        print("  - これは定在波の特性であり、正しい挙動")
        print("  - 総エネルギー E = K + P は保存される")
    else:
        print("⚠️  数値誤差による総エネルギーの変動あり")
        print("   改善策: CFL数を1.0に近づける、または高次差分を使用")
    
    plt.show()

if __name__ == "__main__":
    test_energy_periodicity()