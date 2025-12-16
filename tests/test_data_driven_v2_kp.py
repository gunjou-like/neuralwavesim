"""
Data-Driven v2 Model: Inference Test with K-P Phase Diagram
Uses ModelFactory instead of direct import
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


def test_data_driven_v2_phase_diagram(
    nx=100,
    nt=200,
    L=10.0,
    T=10.0,
    c=1.0,
    save_path='tests/results/data_driven_v2_phase_diagram.png'
):
    """Test Data-Driven v2 model and generate K-P phase diagram"""
    print("=" * 70)
    print("Data-Driven v2: K-P Phase Diagram Test")
    print("=" * 70)
    
    # Check model existence
    model_path = Path('models/checkpoints/data_driven_v2.pth')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("  python training/train_data_driven_v2.py")
        return
    
    # ✅ UIと完全に同じパラメータ
    params = PhysicsParams(
        nx=nx,
        nt=nt,
        c=c,
        dt=0.05,     # UIと同じ
        dx=L / nx,   # UIと同じ
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
    model = ModelFactory.create('data-driven')  # ← 'data-driven-v2' ではなく 'data-driven'
    print(f"✅ Model loaded")
    
    # Run simulation
    print(f"\nRunning simulation...")
    wave_history = model.predict(ic, params)
    print(f"✅ Simulation complete: shape {wave_history.shape}")
    
    # ✅ 初期波形の確認
    print(f"\nInitial Wave Statistics:")
    print(f"  Max: {np.max(wave_history[0]):.4f}")
    print(f"  Min: {np.min(wave_history[0]):.4f}")
    print(f"  Mean: {np.mean(wave_history[0]):.4f}")
    print(f"  Squared sum: {np.sum(wave_history[0]**2):.6f}")
    
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
    
    if E_variation < 5.0:
        status = "✅ Excellent energy conservation"
    elif E_variation < 10.0:
        status = "⚠️  Acceptable energy conservation"
    else:
        status = "❌ Poor energy conservation"
    print(f"  Status: {status}")
    
    # Create figure
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data-Driven: Energy Analysis', fontsize=16, fontweight='bold')
    
    time_points = (np.arange(1, params.nt - 1) * params.dt)
    
    # 1. K-P Phase Diagram
    ax = axes[0, 0]
    scatter = ax.scatter(K_history, P_history, c=np.arange(len(K_history)), 
                        cmap='viridis', s=20, alpha=0.6)
    ax.plot(K_history[0], P_history[0], 'go', markersize=12, 
            label='Start', zorder=5)
    ax.plot(K_history[-1], P_history[-1], 'ro', markersize=12, 
            label='End', zorder=5)
    ax.set_xlabel('Kinetic Energy (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Potential Energy (P)', fontsize=12, fontweight='bold')
    ax.set_title('K-P Phase Diagram', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time Step')
    
    # 2. Energy vs Time
    ax = axes[0, 1]
    ax.plot(time_points, K_history, 'b-', label='Kinetic (K)', linewidth=2)
    ax.plot(time_points, P_history, 'r-', label='Potential (P)', linewidth=2)
    ax.plot(time_points, E_history, 'g--', label='Total (E)', linewidth=2.5)
    ax.axhline(E_mean, color='gray', linestyle=':', linewidth=2, 
               label=f'Mean: {E_mean:.3f}')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy', fontsize=12, fontweight='bold')
    ax.set_title(f'Energy vs Time (Variation: {E_variation:.2f}%)', 
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Wave Evolution (Heatmap)
    ax = axes[1, 0]
    x_grid = np.linspace(0, L, nx)
    t_grid = np.linspace(0, T, nt)
    im = ax.pcolormesh(t_grid, x_grid, wave_history.T, 
                       cmap='RdBu_r', shading='auto')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Wave Evolution', fontweight='bold', fontsize=13)
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    # 4. Energy Ratio K/P
    ax = axes[1, 1]
    ratio = K_history / (P_history + 1e-10)
    ax.plot(time_points, ratio, 'purple', linewidth=2.5)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='K = P')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K/P Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Kinetic/Potential Energy Ratio', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
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
        'E_variation': E_variation
    }


if __name__ == "__main__":
    results = test_data_driven_v2_phase_diagram()
    
    if results:
        print("\n" + "=" * 70)
        print("Test Complete!")
        print("=" * 70)
        print(f"Energy Variation: {results['E_variation']:.2f}%")
        
        if results['E_variation'] < 5.0:
            print("Status: ✅ Excellent")
        elif results['E_variation'] < 10.0:
            print("Status: ⚠️  Acceptable")
        else:
            print("Status: ❌ Needs improvement")