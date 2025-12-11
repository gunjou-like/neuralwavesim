import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition

def calculate_energy_detailed(wave_history, dt, dx, c):
    """
    詳細なエネルギー計算（運動・ポテンシャル・総エネルギー）
    """
    nt, nx = wave_history.shape
    
    kinetic_energy = np.zeros(nt - 2)
    potential_energy = np.zeros(nt - 2)
    total_energy = np.zeros(nt - 2)
    
    for t in range(1, nt - 1):
        # 運動エネルギー: (∂u/∂t)^2
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
        K = 0.5 * np.sum(u_t**2) * dx
        
        # ポテンシャルエネルギー: c^2 * (∂u/∂x)^2
        # 中心差分で空間微分
        u_x = np.zeros(nx)
        u_x[1:-1] = (wave_history[t, 2:] - wave_history[t, :-2]) / (2 * dx)
        # 境界は片側差分
        u_x[0] = (wave_history[t, 1] - wave_history[t, 0]) / dx
        u_x[-1] = (wave_history[t, -1] - wave_history[t, -2]) / dx
        
        P = 0.5 * c**2 * np.sum(u_x**2) * dx
        
        kinetic_energy[t-1] = K
        potential_energy[t-1] = P
        total_energy[t-1] = K + P
    
    return kinetic_energy, potential_energy, total_energy

def test_energy_conservation():
    """差分法でエネルギー保存則を検証"""
    print("=" * 60)
    print("エネルギー保存則の検証")
    print("=" * 60)
    
    # 物理パラメータ（CFL条件を満たす）
    params = PhysicsParams(
        nx=100,
        nt=200,
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=10.0
    )
    
    # CFL数の確認
    print(f"CFL数 (C): {params.courant_number:.4f}")
    print(f"安定性条件: C ≤ 1.0 → {params.courant_number <= 1.0}")
    print()
    
    # 初期条件
    ic = InitialCondition(wave_type="gaussian", center=5.0, width=1.0, height=1.0)
    
    # シミュレーション実行
    solver = WaveSolver(params)
    wave_history = solver.solve(ic)
    
    # エネルギー計算
    K, P, E = calculate_energy_detailed(wave_history, params.dt, params.dx, params.c)
    
    # 統計情報
    E_mean = np.mean(E)
    E_std = np.std(E)
    E_variation = (np.max(E) - np.min(E)) / E_mean * 100
    
    print(f"総エネルギー（統計）:")
    print(f"  平均値: {E_mean:.6f}")
    print(f"  標準偏差: {E_std:.6f} ({E_std/E_mean*100:.2f}%)")
    print(f"  変動範囲: {E_variation:.2f}%")
    print(f"  最大値: {np.max(E):.6f}")
    print(f"  最小値: {np.min(E):.6f}")
    print()
    
    # エネルギー保存の判定
    if E_variation < 1.0:
        print("✅ エネルギー保存則を満たしている（誤差 < 1%）")
    elif E_variation < 5.0:
        print("⚠️  数値誤差あり（誤差 1-5%）")
    else:
        print("❌ エネルギー保存則が破綻（誤差 > 5%）")
    
    print("=" * 60)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 波形の時間発展
    ax = axes[0, 0]
    time_steps = [0, 50, 100, 150, 199]
    for t in time_steps:
        ax.plot(wave_history[t], label=f't={t}', alpha=0.7)
    ax.set_title('波形の時間発展', fontweight='bold')
    ax.set_xlabel('位置 (x)')
    ax.set_ylabel('変位 (u)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) 時空間ヒートマップ
    ax = axes[0, 1]
    im = ax.imshow(wave_history, aspect='auto', cmap='RdBu', 
                   extent=[0, params.L, 0, params.T_max], origin='lower')
    ax.set_title('時空間分布', fontweight='bold')
    ax.set_xlabel('位置 (x)')
    ax.set_ylabel('時刻 (t)')
    plt.colorbar(im, ax=ax, label='変位 (u)')
    
    # (3) エネルギーの時間変化
    ax = axes[1, 0]
    time_axis = np.arange(1, len(E) + 1) * params.dt
    ax.plot(time_axis, K, label='運動エネルギー (K)', alpha=0.7)
    ax.plot(time_axis, P, label='ポテンシャルエネルギー (P)', alpha=0.7)
    ax.plot(time_axis, E, 'k-', label='総エネルギー (E)', linewidth=2)
    ax.axhline(E_mean, color='r', linestyle='--', label='平均値')
    ax.set_title('エネルギーの時間変化', fontweight='bold')
    ax.set_xlabel('時刻 (t)')
    ax.set_ylabel('エネルギー')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (4) 総エネルギーの相対誤差
    ax = axes[1, 1]
    relative_error = (E - E_mean) / E_mean * 100
    ax.plot(time_axis, relative_error, 'r-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(time_axis, -1, 1, alpha=0.2, color='green', label='許容範囲 (±1%)')
    ax.set_title('総エネルギーの相対誤差', fontweight='bold')
    ax.set_xlabel('時刻 (t)')
    ax.set_ylabel('相対誤差 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_conservation_test.png', dpi=150, bbox_inches='tight')
    print("\n✅ 図を保存: energy_conservation_test.png")
    plt.show()
    
    return E_variation

if __name__ == "__main__":
    variation = test_energy_conservation()