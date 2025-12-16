"""
PINNs (Original) Model: K-P Phase Diagram Test
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


def test_pinns_phase_diagram(
    nx=100,
    nt=200,
    L=10.0,
    T=10.0,
    c=1.0,
    save_path='tests/results/pinns_phase_diagram.png'
):
    """Test PINNs (Original) model and generate K-P phase diagram"""
    print("=" * 70)
    print("PINNs (Original): K-P Phase Diagram Test")
    print("=" * 70)
    
    # Check model existence
    model_path = Path('models/checkpoints/wave_pinns.pth')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("  python training/train_pinns.py")
        return
    
    # ✅ UIと完全に同じパラメータ
    params = PhysicsParams(
        nx=nx,
        nt=nt,
        c=c,
        dt=0.05,
        dx=L / nx,
        L=L,
        T_max=T
    )
    
    # ✅ UIと完全に同じ初期条件
    ic = InitialCondition(
        wave_type="gaussian",
        center=L / 2,
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
    model = ModelFactory.create('pinns')
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
    
    # Compute energies
    print(f"\nComputing energies...")
    K_history, P_history, E_history = compute_energy(wave_history, params)
    
    # ✅ UIと同じ統計計算
    E_mean = np.mean(E_history)
    E_std = np.std(E_history)
    E_min = np.min(E_history)
    E_max = np.max(E_history)
    E_variation = (E_max - E_min) / E_mean * 100
    
    print(f"\n📊 Energy Statistics:")
    print(f"  Mean Total Energy: {E_mean:.6f}")
    print(f"  Std Deviation: {E_std:.6f}")
    print(f"  Min Energy: {E_min:.6f}")
    print(f"  Max Energy: {E_max:.6f}")
    print(f"  Energy Variation (Range/Mean): {E_variation:.2f}%")
    print(f"  Energy Variation (Std/Mean): {(E_std/E_mean)*100:.2f}%")
    
    # ✅ デバッグ情報
    print(f"\n🔍 Debug:")
    print(f"  Number of energy samples: {len(E_history)}")
    print(f"  First 5 energies: {E_history[:5]}")
    print(f"  Last 5 energies: {E_history[-5:]}")
    
    if E_variation < 5.0:
        status = "✅ Excellent energy conservation"
    elif E_variation < 10.0:
        status = "⚠️  Acceptable energy conservation"
    else:
        status = "❌ Poor energy conservation"
    print(f"\n  Status: {status}")
    
    # ✅ K-P Phase Diagram のみ作成
    print(f"\nGenerating K-P phase diagram...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    time_points = np.arange(1, params.nt - 1)
    
    # K-P Phase Diagram
    scatter = ax.scatter(K_history, P_history, c=time_points, 
                        cmap='plasma', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Start and End points
    ax.plot(K_history[0], P_history[0], 'go', markersize=20, 
            label='Start (t=0)', zorder=5, markeredgecolor='white', markeredgewidth=3)
    ax.plot(K_history[-1], P_history[-1], 'ro', markersize=20, 
            label=f'End (t={params.nt-2})', zorder=5, markeredgecolor='white', markeredgewidth=3)
    
    # Labels and title
    ax.set_xlabel('Kinetic Energy (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Potential Energy (P)', fontsize=14, fontweight='bold')
    ax.set_title(f'PINNs (Original): K-P Phase Diagram',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Time Step')
    cbar.set_label('Time Step', fontsize=12, fontweight='bold')
    
    # Add text box with statistics
    textstr = '\n'.join([
        f'Statistics:',
        f'Mean E: {E_mean:.2f}',
        f'Std E: {E_std:.2f}',
        f'Min E: {E_min:.2f}',
        f'Max E: {E_max:.2f}',
        f'Variation: {E_variation:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
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
        'E_variation': E_variation,
        'wave_history': wave_history
    }


if __name__ == "__main__":
    results = test_pinns_phase_diagram()
    
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
            print("Status: ❌ Poor")