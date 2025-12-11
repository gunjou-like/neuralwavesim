"""
Verify improved PINNs model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from models.pinns_v2 import PINNsModel_v2
from models.factory import ModelFactory  # ★ 修正: factory経由で取得
from core.config import PhysicsParams, InitialCondition

def compare_pinns_v1_v2():
    """Compare original and improved PINNs"""
    print("=" * 70)
    print("PINNs v1 vs v2 Comparison")
    print("=" * 70)
    
    # Parameters
    params = PhysicsParams(
        nx=100,
        nt=200,
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=10.0
    )
    
    ic = InitialCondition(
        wave_type="gaussian",
        center=5.0,
        width=1.0,
        height=1.0
    )
    
    # Models
    physics_model = ModelFactory.create("physics")  # ★ 修正
    pinns_v2_model = PINNsModel_v2(model_path='models/pinns_v2.pth')
    
    # Predictions
    print("\nRunning simulations...")
    wave_physics = physics_model.predict(ic, params)
    wave_pinns_v2 = pinns_v2_model.predict(ic, params)
    
    # Energy analysis
    def calculate_energy(wave_history, dt, dx, c):
        energies = []
        for t in range(1, len(wave_history) - 1):
            u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * dt)
            u_x = np.gradient(wave_history[t], dx)
            
            K = 0.5 * np.sum(u_t**2) * dx
            P = 0.5 * c**2 * np.sum(u_x**2) * dx
            E = K + P
            energies.append(E)
        return np.array(energies)
    
    E_physics = calculate_energy(wave_physics, params.dt, params.dx, params.c)
    E_pinns_v2 = calculate_energy(wave_pinns_v2, params.dt, params.dx, params.c)
    
    # Energy variation
    E_var_physics = (np.max(E_physics) - np.min(E_physics)) / np.mean(E_physics) * 100
    E_var_pinns_v2 = (np.max(E_pinns_v2) - np.min(E_pinns_v2)) / np.mean(E_pinns_v2) * 100
    
    print(f"\nEnergy Conservation:")
    print(f"  Physics-based: {E_var_physics:.2f}%")
    print(f"  PINNs v2:      {E_var_pinns_v2:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy comparison
    ax = axes[0, 0]
    time_points = np.arange(1, params.nt - 1) * params.dt
    ax.plot(time_points, E_physics, 'b-', linewidth=2, label='Physics-based', alpha=0.7)
    ax.plot(time_points, E_pinns_v2, 'r--', linewidth=2, label='PINNs v2', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Snapshots at t=5s
    ax = axes[0, 1]
    t_idx = int(5.0 / params.dt)
    x = np.linspace(0, params.L, params.nx)
    ax.plot(x, wave_physics[t_idx], 'b-', linewidth=2, label='Physics-based', alpha=0.7)
    ax.plot(x, wave_pinns_v2[t_idx], 'r--', linewidth=2, label='PINNs v2', alpha=0.7)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement u')
    ax.set_title(f'Waveform at t={t_idx*params.dt:.1f}s', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heatmap: Physics
    ax = axes[1, 0]
    t = np.arange(params.nt) * params.dt
    X, T = np.meshgrid(x, t)
    im = ax.pcolormesh(X, T, wave_physics, cmap='RdBu_r', shading='auto')
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Time t (s)')
    ax.set_title('Physics-based', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Heatmap: PINNs v2
    ax = axes[1, 1]
    im = ax.pcolormesh(X, T, wave_pinns_v2, cmap='RdBu_r', shading='auto')
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Time t (s)')
    ax.set_title('PINNs v2', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('pinns_v2_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Comparison saved: pinns_v2_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_pinns_v1_v2()