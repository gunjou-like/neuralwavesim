import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition

def generate_training_data_snapshots(
    n_snapshots: int = 8,
    save_path: str = "training_data_snapshots.png"
):
    """
    Generate time evolution snapshots of training data
    
    Args:
        n_snapshots: Number of snapshots (default: 8)
        save_path: Output file path
    """
    print("=" * 70)
    print("Training Data Visualization: Time Evolution Snapshots")
    print("=" * 70)
    
    # Parameters (same as training)
    params = PhysicsParams(
        nx=100,
        nt=200,
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=10.0
    )
    
    # Initial condition (standard Gaussian pulse)
    ic = InitialCondition(
        wave_type="gaussian",
        center=5.0,
        width=1.0,
        height=1.0
    )
    
    print(f"\nParameters:")
    print(f"  Spatial grid points: nx = {params.nx}")
    print(f"  Time steps: nt = {params.nt}")
    print(f"  Wave speed: c = {params.c:.2f} m/s")
    print(f"  Time step: dt = {params.dt:.3f} s")
    print(f"  Spatial step: dx = {params.dx:.3f} m")
    print(f"  CFL number: C = {params.courant_number:.3f}")
    
    print(f"\nInitial Condition:")
    print(f"  Wave type: {ic.wave_type}")
    print(f"  Center position: x₀ = {ic.center:.2f} m")
    print(f"  Pulse width: σ = {ic.width:.2f} m")
    print(f"  Amplitude: h = {ic.height:.2f}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver = WaveSolver(params)
    wave_history = solver.solve(ic)
    print(f"✅ Completed: {wave_history.shape}")
    
    # Select snapshot time steps
    # One period = 2L/c = 20 s
    period_steps = int(2 * params.L / params.c / params.dt)  # 400 steps
    
    # Adjust based on actual number of steps
    max_steps = params.nt
    
    if period_steps > max_steps:
        # Divide full range equally
        snapshot_indices = np.linspace(0, max_steps - 1, n_snapshots, dtype=int)
        print(f"\n⚠️  One period ({period_steps} steps) > Total steps ({max_steps})")
        print(f"   Dividing full range into {n_snapshots} parts")
    else:
        # Divide one period equally
        snapshot_indices = np.linspace(0, period_steps - 1, n_snapshots, dtype=int)
        snapshot_indices = snapshot_indices[snapshot_indices < max_steps]
        print(f"\nOne period = {period_steps} steps ({period_steps * params.dt:.2f} s)")
        print(f"Snapshot interval: {period_steps // n_snapshots} steps")
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    x = np.linspace(0, params.L, params.nx)
    
    # Each snapshot
    for i, t_idx in enumerate(snapshot_indices):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Plot waveform
        ax.plot(x, wave_history[t_idx], 'b-', linewidth=2, label='u(x,t)')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(x, 0, wave_history[t_idx], alpha=0.2)
        
        # Boundary lines
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(params.L, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Title
        time_value = t_idx * params.dt
        ax.set_title(f't = {time_value:.2f} s (step {t_idx})', fontweight='bold', fontsize=11)
        
        # Axis labels
        ax.set_xlabel('Position x (m)', fontsize=10)
        ax.set_ylabel('Displacement u', fontsize=10)
        
        # Unified axis range
        ax.set_xlim([0, params.L])
        ax.set_ylim([-1.2 * ic.height, 1.2 * ic.height])
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    # Overall title
    fig.suptitle(
        f'Training Data Time Evolution Snapshots\n'
        f'(Initial Condition: Gaussian pulse, center={ic.center}, width={ic.width}, height={ic.height})',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved: {save_path}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Max displacement: {np.max(np.abs(wave_history)):.4f}")
    print(f"  Min displacement: {np.min(wave_history):.4f}")
    print(f"  Mean displacement: {np.mean(wave_history):.4f}")
    print(f"  Std deviation: {np.std(wave_history):.4f}")
    
    # Energy calculation
    if params.nt >= 3:
        energies = []
        for t in range(1, params.nt - 1):
            u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * params.dt)
            u_x = np.gradient(wave_history[t], params.dx)
            
            K = 0.5 * np.sum(u_t**2) * params.dx
            P = 0.5 * params.c**2 * np.sum(u_x**2) * params.dx
            E = K + P
            energies.append(E)
        
        energies = np.array(energies)
        E_mean = np.mean(energies)
        E_variation = (np.max(energies) - np.min(energies)) / E_mean * 100
        
        print(f"\nEnergy:")
        print(f"  Mean total energy: {E_mean:.4f}")
        print(f"  Variation: {E_variation:.2f}%")
        
        if E_variation < 5.0:
            print(f"  Evaluation: ✅ Excellent (variation < 5%)")
        elif E_variation < 10.0:
            print(f"  Evaluation: ⚠️  Acceptable (variation < 10%)")
        else:
            print(f"  Evaluation: ❌ Needs improvement (variation ≥ 10%)")
    
    plt.show()
    
    return wave_history, snapshot_indices

def visualize_training_data_structure():
    """
    Visualize training data structure
    (Input → Output correspondence)
    """
    print("\n" + "=" * 70)
    print("Training Data Structure Visualization")
    print("=" * 70)
    
    # Parameters
    params = PhysicsParams(nx=100, nt=200, c=1.0, dt=0.05, dx=0.1, L=10.0, T_max=10.0)
    ic = InitialCondition(wave_type="gaussian", center=5.0, width=1.0, height=1.0)
    
    # Simulation
    solver = WaveSolver(params)
    wave_history = solver.solve(ic)
    
    # Example training data pair (at t=50)
    t_example = 50
    
    # Input: current waveform u(x, t)
    input_data = wave_history[t_example]
    
    # Output: next time step waveform u(x, t+Δt)
    output_data = wave_history[t_example + 1]
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.linspace(0, params.L, params.nx)
    
    # (1) Input data
    ax = axes[0, 0]
    ax.plot(x, input_data, 'b-', linewidth=2, marker='o', markersize=3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f'Input: u(x, t={t_example * params.dt:.2f}s)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement u')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Shape: ({params.nx},)', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # (2) Output data
    ax = axes[0, 1]
    ax.plot(x, output_data, 'r-', linewidth=2, marker='s', markersize=3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f'Output: u(x, t={(t_example+1) * params.dt:.2f}s)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement u')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Shape: ({params.nx},)', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # (3) Input and output overlay
    ax = axes[1, 0]
    ax.plot(x, input_data, 'b-', linewidth=2, label=f't = {t_example * params.dt:.2f}s (Input)', alpha=0.7)
    ax.plot(x, output_data, 'r--', linewidth=2, label=f't = {(t_example+1) * params.dt:.2f}s (Output)', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Input vs Output Comparison', fontweight='bold', fontsize=12)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement u')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # (4) Difference (change amount)
    ax = axes[1, 1]
    diff = output_data - input_data
    ax.plot(x, diff, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(x, 0, diff, alpha=0.3, color='green')
    ax.set_title('Change: Δu = u(t+Δt) - u(t)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Δu')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max |Δu|: {np.max(np.abs(diff)):.4f}', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    fig.suptitle('Training Data Structure (Input → Output)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_data_structure.png', dpi=150, bbox_inches='tight')
    print("✅ Figure saved: training_data_structure.png")
    plt.show()

if __name__ == "__main__":
    # (1) Time evolution snapshots
    wave_history, snapshot_indices = generate_training_data_snapshots(
        n_snapshots=8,
        save_path="training_data_snapshots.png"
    )
    
    # (2) Data structure visualization
    visualize_training_data_structure()
    
    print("\n" + "=" * 70)
    print("✅ All visualizations completed")
    print("=" * 70)