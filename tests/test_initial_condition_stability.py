import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition

def calculate_energy_variation(wave_history, dt, dx, c):
    """エネルギー変動率を計算"""
    nt, nx = wave_history.shape
    
    if nt < 3:
        return float('inf')
    
    energy = np.zeros(nt - 2)
    
    for t in range(1, nt - 1):
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
        u_x = np.gradient(wave_history[t], dx)
        
        K = 0.5 * np.sum(u_t**2) * dx
        P = 0.5 * c**2 * np.sum(u_x**2) * dx
        energy[t-1] = K + P
    
    E_mean = np.mean(energy)
    if E_mean == 0:
        return float('inf')
    
    E_variation = (np.max(energy) - np.min(energy)) / E_mean * 100
    
    # NaN や Inf のチェック
    if np.isnan(E_variation) or np.isinf(E_variation):
        return float('inf')
    
    return E_variation

def test_initial_condition_stability():
    """様々な初期条件での安定性テスト"""
    print("=" * 70)
    print("初期条件による安定性テスト")
    print("=" * 70)
    
    # 基本パラメータ
    params = PhysicsParams(
        nx=100,
        nt=200,
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=10.0
    )
    
    print(f"CFL数: {params.courant_number:.4f}\n")
    
    # テストケース
    test_cases = []
    
    # 1. 中心位置を変化
    for center in [1.0, 2.5, 5.0, 7.5, 9.0]:
        test_cases.append({
            'name': f'center={center}',
            'ic': InitialCondition(wave_type="gaussian", center=center, width=1.0, height=1.0)
        })
    
    # 2. 幅を変化
    for width in [0.3, 0.5, 1.0, 2.0, 3.0]:
        test_cases.append({
            'name': f'width={width}',
            'ic': InitialCondition(wave_type="gaussian", center=5.0, width=width, height=1.0)
        })
    
    # 3. 高さを変化
    for height in [0.5, 1.0, 2.0, 5.0, 10.0]:
        test_cases.append({
            'name': f'height={height}',
            'ic': InitialCondition(wave_type="gaussian", center=5.0, width=1.0, height=height)
        })
    
    # 結果格納
    results = []
    
    for test in test_cases:
        try:
            solver = WaveSolver(params)
            wave_history = solver.solve(test['ic'])
            
            # NaN/Inf チェック
            if np.isnan(wave_history).any() or np.isinf(wave_history).any():
                variation = float('inf')
                status = "❌ 発散（NaN/Inf）"
                max_amp = float('nan')
            else:
                variation = calculate_energy_variation(wave_history, params.dt, params.dx, params.c)
                max_amp = np.max(np.abs(wave_history))
                
                if variation < 5.0:
                    status = "✅ 安定"
                elif variation < 20.0:
                    status = "⚠️  不安定"
                else:
                    status = "❌ 発散"
            
            results.append({
                'name': test['name'],
                'variation': variation,
                'status': status,
                'max_amplitude': max_amp
            })
            
            print(f"{test['name']:20s}: 変動率 {variation:6.2f}%  最大振幅 {max_amp:8.4f}  {status}")
        
        except Exception as e:
            print(f"{test['name']:20s}: ❌ エラー - {e}")
            results.append({
                'name': test['name'],
                'variation': float('inf'),
                'status': f"❌ エラー: {e}",
                'max_amplitude': float('nan')
            })
    
    print("\n" + "=" * 70)
    print("結果サマリー")
    print("=" * 70)
    
    stable_count = sum(1 for r in results if '✅' in r['status'])
    unstable_count = sum(1 for r in results if '⚠️' in r['status'])
    diverged_count = sum(1 for r in results if '❌' in r['status'])
    
    print(f"安定: {stable_count}/{len(results)}")
    print(f"不安定: {unstable_count}/{len(results)}")
    print(f"発散: {diverged_count}/{len(results)}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 中心位置依存性
    ax = axes[0]
    center_results = [r for r in results if 'center=' in r['name']]
    centers = [float(r['name'].split('=')[1]) for r in center_results]
    variations = [r['variation'] if r['variation'] != float('inf') else 100 for r in center_results]
    
    ax.plot(centers, variations, 'o-', linewidth=2, markersize=8)
    ax.axhline(5.0, color='orange', linestyle='--', label='許容閾値 (5%)')
    ax.axhline(20.0, color='red', linestyle='--', label='発散閾値 (20%)')
    ax.set_xlabel('中心位置 (x)', fontsize=12)
    ax.set_ylabel('エネルギー変動率 (%)', fontsize=12)
    ax.set_title('中心位置依存性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 50])
    
    # 幅依存性
    ax = axes[1]
    width_results = [r for r in results if 'width=' in r['name']]
    widths = [float(r['name'].split('=')[1]) for r in width_results]
    variations = [r['variation'] if r['variation'] != float('inf') else 100 for r in width_results]
    
    ax.plot(widths, variations, 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(5.0, color='orange', linestyle='--', label='許容閾値 (5%)')
    ax.axhline(20.0, color='red', linestyle='--', label='発散閾値 (20%)')
    ax.set_xlabel('幅 (σ)', fontsize=12)
    ax.set_ylabel('エネルギー変動率 (%)', fontsize=12)
    ax.set_title('パルス幅依存性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 50])
    
    # 高さ依存性
    ax = axes[2]
    height_results = [r for r in results if 'height=' in r['name']]
    heights = [float(r['name'].split('=')[1]) for r in height_results]
    variations = [r['variation'] if r['variation'] != float('inf') else 100 for r in height_results]
    
    ax.plot(heights, variations, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(5.0, color='orange', linestyle='--', label='許容閾値 (5%)')
    ax.axhline(20.0, color='red', linestyle='--', label='発散閾値 (20%)')
    ax.set_xlabel('高さ (h)', fontsize=12)
    ax.set_ylabel('エネルギー変動率 (%)', fontsize=12)
    ax.set_title('振幅依存性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 50])
    
    plt.tight_layout()
    plt.savefig('initial_condition_stability.png', dpi=150)
    print("\n✅ 図を保存: initial_condition_stability.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    results = test_initial_condition_stability()