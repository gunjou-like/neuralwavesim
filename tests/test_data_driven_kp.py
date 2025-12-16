"""
Data-Driven (Original) Model: K-P Phase Diagram Test
UIと完全に同じ条件でテスト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from core.config import PhysicsParams, InitialCondition
from models.factory import ModelFactory


def compute_energy(wave_history, params):
    """
    ✅ UIと完全に同一の実装
    """
    energies = []
    kinetic_energies = []
    potential_energies = []
    
    for i in range(1, params.nt - 1):
        u_t = (wave_history[i+1] - wave_history[i-1]) / (2 * params.dt)
        u_x = np.gradient(wave_history[i], params.dx)
        
        K = 0.5 * np.sum(u_t**2) * params.dx
        P = 0.5 * params.c**2 * np.sum(u_x**2) * params.dx
        E = K + P
        
        kinetic_energies.append(K)
        potential_energies.append(P)
        energies.append(E)
    
    energies = np.array(energies)
    kinetic_energies = np.array(kinetic_energies)
    potential_energies = np.array(potential_energies)
    
    return kinetic_energies, potential_energies, energies


def test_data_driven_phase_diagram(
    nx=100,
    nt=200,
    L=10.0,
    T=10.0,
    c=1.0,
    save_path='tests/results/data_driven_phase_diagram.png'
):
    """Test Data-Driven (Original) model and generate K-P phase diagram"""
    print("=" * 70)
    print("Data-Driven (Original): K-P Phase Diagram Test")
    print("=" * 70)
    
    # Check model existence
    model_path = Path('models/checkpoints/wave_model.pth')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("  python training/train_data_driven.py")
        return
    
    # ✅ UIと完全に同じパラメータ
    params = PhysicsParams(
        nx=nx,
        nt=nt,
        c=c,
        dt=0.05,     # UIと同じ
        dx=L / nx,   # UIと同じ (10.0 / 100 = 0.1)
        L=L,
        T_max=T
    )
    
    # ✅ UIと完全に同じ初期条件
    ic = InitialCondition(
        wave_type="gaussian",
        center=L / 2,    # 5.0
        width=1.0,
        height=1.0
    )
    
    print(f"\nSimulation Parameters:")
    print(f"  Domain: L = {L} m, T = {T} s")
    print(f"  Grid: nx = {nx}, nt = {nt}")
    print(f"  Steps: dx = {params.dx:.4f} m, dt = {params.dt:.4f} s")
    print(f"  Wave speed: c = {c} m/s")
    print(f"\nInitial Condition:")
    print(f"  Type: {ic.wave_type}")
    print(f"  Center: {ic.center} m")
    print(f"  Width: {ic.width} m")
    print(f"  Height: {ic.height}")
    
    # Load model
    print(f"\nLoading model...")
    model = ModelFactory.create('data-driven')
    print(f"✅ Model loaded: {model_path}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    wave_history = model.predict(ic, params)
    print(f"✅ Simulation complete: shape {wave_history.shape}")
    
    # ✅ 初期波形の確認
    print(f"\nInitial Wave Statistics:")
    print(f"  Max: {np.max(wave_history[0]):.6f}")
    print(f"  Min: {np.min(wave_history[0]):.6f}")
    print(f"  Mean: {np.mean(wave_history[0]):.6f}")
    print(f"  Squared sum: {np.sum(wave_history[0]**2):.6f}")
    
    # ✅ 波の振幅の時間変化を確認
    print(f"\nWave Amplitude Over Time:")
    print(f"  t=0:   max={np.max(np.abs(wave_history[0])):.6f}")
    print(f"  t=50:  max={np.max(np.abs(wave_history[50])):.6f}")
    print(f"  t=100: max={np.max(np.abs(wave_history[100])):.6f}")
    print(f"  t=150: max={np.max(np.abs(wave_history[150])):.6f}")
    print(f"  t=199: max={np.max(np.abs(wave_history[199])):.6f}")
    
    # Compute energies
    print(f"\nComputing energies...")
    K_history, P_history, E_history = compute_energy(wave_history, params)
    
    # ✅ UIと同じ統計計算
    E_mean = np.mean(E_history)
    E_std = np.std(E_history)
    E_min = np.min(E_history)
    E_max = np.max(E_history)
    E_variation = (E_max - E_min) / E_mean * 100  # ← UIと同じ式
    
    print(f"\n📊 Energy Statistics:")
    print(f"  Mean Total Energy: {E_mean:.6f}")
    print(f"  Std Deviation: {E_std:.6f}")
    print(f"  Min Energy: {E_min:.6f}")
    print(f"  Max Energy: {E_max:.6f}")
    print(f"  Energy Variation (Range/Mean): {E_variation:.2f}%")
    print(f"  Energy Variation (Std/Mean): {(E_std/E_mean)*100:.2f}%")
    
    # ✅ UIと同じデバッグ情報
    print(f"\n🔍 Debug:")
    print(f"  Number of energy samples: {len(E_history)}")
    print(f"  First 5 energies: {E_history[:5]}")
    print(f"  Last 5 energies: {E_history[-5:]}")
    
    # ✅ 手動計算の検証
    print(f"\n✅ Manual Calculation at t=1:")
    u_t_manual = (wave_history[2] - wave_history[0]) / (2 * params.dt)
    u_x_manual = np.gradient(wave_history[1], params.dx)
    K_manual = 0.5 * np.sum(u_t_manual**2) * params.dx
    P_manual = 0.5 * params.c**2 * np.sum(u_x_manual**2) * params.dx
    E_manual = K_manual + P_manual
    print(f"  K(t=1): {K_manual:.6f}")
    print(f"  P(t=1): {P_manual:.6f}")
    print(f"  E(t=1): {E_manual:.6f}")
    print(f"  Match with E_history[0]? {np.isclose(E_history[0], E_manual)}")
    
    # ✅ UIとの比較
    print(f"\n🎯 Comparison with UI:")
    print(f"  Expected E_mean (UI): 22.656440")
    print(f"  Actual E_mean:        {E_mean:.6f}")
    print(f"  Expected Variation:   385.24%")
    print(f"  Actual Variation:     {E_variation:.2f}%")
    print(f"  Match? {np.isclose(E_variation, 385.24, atol=1.0)}")
    
    if E_variation < 5.0:
        status = "✅ Excellent energy conservation"
    elif E_variation < 10.0:
        status = "⚠️  Acceptable energy conservation"
    else:
        status = "❌ Poor energy conservation (expected for data-driven)"
    print(f"\n  Status: {status}")
    
    # Create figure
    print(f"\nGenerating plots...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Data-Driven (Original): Energy Analysis', 
                 fontsize=16, fontweight='bold')
    
    time_points = (np.arange(1, params.nt - 1) * params.dt)
    
    # 1. K-P Phase Diagram (large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(K_history, P_history, c=np.arange(len(K_history)), 
                         cmap='viridis', s=30, alpha=0.6)
    ax1.plot(K_history[0], P_history[0], 'go', markersize=15, 
            label='Start', zorder=5, markeredgecolor='white', markeredgewidth=2)
    ax1.plot(K_history[-1], P_history[-1], 'ro', markersize=15, 
            label='End', zorder=5, markeredgecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Kinetic Energy (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Potential Energy (P)', fontsize=13, fontweight='bold')
    ax1.set_title('K-P Phase Diagram (Diverging ❌)', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Time Step')
    
    # 2. Energy vs Time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_points, K_history, 'b-', label='Kinetic (K)', linewidth=2, alpha=0.8)
    ax2.plot(time_points, P_history, 'r-', label='Potential (P)', linewidth=2, alpha=0.8)
    ax2.plot(time_points, E_history, 'g-', label='Total (E)', linewidth=3)
    ax2.axhline(E_mean, color='gray', linestyle=':', linewidth=2, 
               label=f'Mean: {E_mean:.1f}')
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Energy', fontsize=11, fontweight='bold')
    ax2.set_title(f'Energy (Var: {E_variation:.1f}%)', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy (Log Scale)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.semilogy(time_points, E_history, 'g-', linewidth=2.5)
    ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Energy (log scale)', fontsize=11, fontweight='bold')
    ax3.set_title('Energy Growth (Exponential)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Wave Evolution (Heatmap)
    ax4 = fig.add_subplot(gs[2, 0:2])
    x_grid = np.linspace(0, L, nx)
    t_grid = np.linspace(0, T, nt)
    
    # Clip extreme values for visualization
    wave_clipped = np.clip(wave_history.T, -2, 2)
    im = ax4.pcolormesh(t_grid, x_grid, wave_clipped, 
                        cmap='RdBu_r', shading='auto', vmin=-2, vmax=2)
    ax4.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Position (m)', fontsize=11, fontweight='bold')
    ax4.set_title('Wave Evolution (clipped to ±2)', fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax4, label='Amplitude')
    
    # 5. K/P Ratio
    ax5 = fig.add_subplot(gs[2, 2])
    ratio = K_history / (P_history + 1e-10)
    ax5.plot(time_points, ratio, 'purple', linewidth=2.5)
    ax5.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='K = P')
    ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('K/P Ratio', fontsize=11, fontweight='bold')
    ax5.set_title('Energy Ratio', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, max(5, np.percentile(ratio, 95))])  # Limit y-axis for visibility
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Phase diagram saved: {save_path}")
    
    plt.show()
    
    return {
        'K': K_history,
        'P': P_history,
        'E': E_history,
        'E_mean': E_mean,
        'E_std': E_std,
        'E_variation': E_variation,
        'wave_history': wave_history
    }


if __name__ == "__main__":
    results = test_data_driven_phase_diagram()
    
    if results:
        print("\n" + "=" * 70)
        print("Test Complete!")
        print("=" * 70)
        print(f"Energy Variation: {results['E_variation']:.2f}%")
        print(f"Expected (UI):    385.24%")
        print(f"Difference:       {abs(results['E_variation'] - 385.24):.2f}%")
        
        if abs(results['E_variation'] - 385.24) < 5.0:
            print("\n✅ Results match UI perfectly!")
        else:
            print("\n⚠️  Results differ from UI")
        
        if results['E_variation'] < 5.0:
            print("Status: ✅ Excellent")
        elif results['E_variation'] < 10.0:
            print("Status: ⚠️  Acceptable")
        else:
            print("Status: ❌ Poor (expected for data-driven model)")