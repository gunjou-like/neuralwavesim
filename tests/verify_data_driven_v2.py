"""
Verify Data-Driven Model v2 Improvements
Compare v1 vs v2 energy conservation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from core.config import PhysicsParams, InitialCondition
from models.factory import ModelFactory


def calculate_energy(wave_history, params):
    """Calculate energy at each time step"""
    energies = []
    kinetic = []
    potential = []
    
    for t in range(1, params.nt - 1):
        # Time derivative
        u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * params.dt)
        
        # Spatial derivative
        u_x = np.gradient(wave_history[t], params.dx)
        
        # Kinetic energy
        K = 0.5 * np.sum(u_t**2) * params.dx
        
        # Potential energy
        P = 0.5 * params.c**2 * np.sum(u_x**2) * params.dx
        
        # Total energy
        E = K + P
        
        energies.append(E)
        kinetic.append(K)
        potential.append(P)
    
    return np.array(energies), np.array(kinetic), np.array(potential)


def compare_data_driven_models():
    """Compare Data-Driven v1 vs v2"""
    print("=" * 70)
    print("Data-Driven v1 vs v2 Comparison")
    print("=" * 70)
    
    # Parameters - ★ 訓練データと同じ条件に変更
    params = PhysicsParams(
        nx=100,
        nt=100,  # ★ 200 → 100 に変更
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=5.0  # ★ 10.0 → 5.0 に変更
    )
    
    # Initial condition
    ic = InitialCondition(
        wave_type="gaussian",
        center=5.0,
        width=1.0,
        height=1.0
    )
    
    print(f"\nTest Configuration:")
    print(f"  Grid: {params.nx} x {params.nt}")
    print(f"  Time: [0, {params.T_max}] s")
    print(f"  Initial condition: Gaussian pulse")
    print(f"  Center: {ic.center}, Width: {ic.width}, Height: {ic.height}")
    
    # Run simulations
    print(f"\nRunning simulations...")
    
    models = {}
    wave_histories = {}
    
    # Physics-based (reference)
    print("  1. Physics-based (reference)...")
    models['physics'] = ModelFactory.create('physics')
    wave_histories['physics'] = models['physics'].predict(ic, params)
    
    # Data-Driven v1 (if exists)
    v1_path = Path('models/data_driven.pth')
    if v1_path.exists():
        print("  2. Data-Driven v1...")
        models['v1'] = ModelFactory.create('data-driven')
        wave_histories['v1'] = models['v1'].predict(ic, params)
    else:
        print("  2. Data-Driven v1... ⚠️ Model not found, skipping")
    
    # Data-Driven v2
    v2_path = Path('models/data_driven_v2.pth')
    if v2_path.exists():
        print("  3. Data-Driven v2...")
        models['v2'] = ModelFactory.create('data-driven-v2')
        wave_histories['v2'] = models['v2'].predict(ic, params)
    else:
        print("  3. Data-Driven v2... ❌ Model not found!")
        print("     Please train: python training/train_data_driven_v2.py")
        return
    
    # Calculate energies
    print(f"\nCalculating energies...")
    energies = {}
    kinetic = {}
    potential = {}
    
    for name, wave_history in wave_histories.items():
        E, K, P = calculate_energy(wave_history, params)
        energies[name] = E
        kinetic[name] = K
        potential[name] = P
    
    # Energy conservation metrics
    print(f"\n" + "=" * 70)
    print("Energy Conservation:")
    print("=" * 70)
    
    for name in energies.keys():
        E = energies[name]
        E_mean = np.mean(E)
        E_std = np.std(E)
        E_variation = (np.max(E) - np.min(E)) / E_mean * 100
        
        status = "✅" if E_variation < 5.0 else "⚠️" if E_variation < 10.0 else "❌"
        
        label = {
            'physics': 'Physics-based',
            'v1': 'Data-Driven v1',
            'v2': 'Data-Driven v2'
        }.get(name, name)
        
        print(f"  {label:20s}: {E_variation:5.2f}% {status}")
    
    # Visualization
    print(f"\nGenerating comparison plots...")
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 3 rows x 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Wave evolution heatmaps
    for i, (name, wave_history) in enumerate(wave_histories.items()):
        ax = fig.add_subplot(gs[0, i])
        
        t = np.arange(params.nt) * params.dt
        x = np.linspace(0, params.L, params.nx)
        
        im = ax.pcolormesh(x, t, wave_history, cmap='RdBu_r', shading='auto')
        
        label = {
            'physics': 'Physics-based (Reference)',
            'v1': 'Data-Driven v1',
            'v2': 'Data-Driven v2 ⭐'
        }.get(name, name)
        
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Time t (s)')
        plt.colorbar(im, ax=ax, label='u')
    
    # Row 2: Energy evolution
    ax = fig.add_subplot(gs[1, :])
    
    colors = {'physics': 'blue', 'v1': 'orange', 'v2': 'green'}
    linestyles = {'physics': '-', 'v1': '--', 'v2': '-'}
    linewidths = {'physics': 2, 'v1': 2, 'v2': 3}
    
    time_points = np.arange(1, params.nt - 1) * params.dt
    
    for name in energies.keys():
        label = {
            'physics': 'Physics-based',
            'v1': 'Data-Driven v1',
            'v2': 'Data-Driven v2 ⭐'
        }.get(name, name)
        
        ax.plot(
            time_points,
            energies[name],
            color=colors.get(name, 'gray'),
            linestyle=linestyles.get(name, '-'),
            linewidth=linewidths.get(name, 2),
            label=label
        )
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy Conservation Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Energy components (v2 only)
    if 'v2' in energies:
        # Kinetic vs Potential
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(time_points, kinetic['v2'], 'r-', linewidth=2, label='Kinetic')
        ax.plot(time_points, potential['v2'], 'b-', linewidth=2, label='Potential')
        ax.plot(time_points, energies['v2'], 'g--', linewidth=2, label='Total')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Components (v2)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy variation over time
        ax = fig.add_subplot(gs[2, 1])
        E_v2 = energies['v2']
        E_mean = np.mean(E_v2)
        variation = (E_v2 - E_mean) / E_mean * 100
        ax.plot(time_points, variation, 'g-', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy Variation (%)')
        ax.set_title('Energy Variation from Mean (v2)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Amplitude comparison
        ax = fig.add_subplot(gs[2, 2])
        for name in wave_histories.keys():
            max_amp = np.max(np.abs(wave_histories[name]), axis=1)
            label = {
                'physics': 'Physics-based',
                'v1': 'Data-Driven v1',
                'v2': 'Data-Driven v2 ⭐'
            }.get(name, name)
            ax.plot(
                np.arange(params.nt) * params.dt,
                max_amp,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=linewidths.get(name, 2),
                label=label
            )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Max Amplitude')
        ax.set_title('Amplitude Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.savefig('data_driven_v2_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Comparison saved: data_driven_v2_comparison.png")
    plt.close()
    
    # Summary
    print(f"\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if 'v1' in energies and 'v2' in energies:
        E_v1 = energies['v1']
        E_v2 = energies['v2']
        
        var_v1 = (np.max(E_v1) - np.min(E_v1)) / np.mean(E_v1) * 100
        var_v2 = (np.max(E_v2) - np.min(E_v2)) / np.mean(E_v2) * 100
        
        improvement = (var_v1 - var_v2) / var_v1 * 100
        
        print(f"  Data-Driven v1: {var_v1:.2f}% energy variation")
        print(f"  Data-Driven v2: {var_v2:.2f}% energy variation")
        print(f"  Improvement: {improvement:.1f}%")
        
        if var_v2 < 5.0:
            print(f"\n✅ Data-Driven v2 achieves excellent energy conservation!")
        elif var_v2 < 10.0:
            print(f"\n⚠️  Data-Driven v2 shows acceptable energy conservation")
        else:
            print(f"\n❌ Data-Driven v2 needs further improvement")
    
    elif 'v2' in energies:
        E_v2 = energies['v2']
        var_v2 = (np.max(E_v2) - np.min(E_v2)) / np.mean(E_v2) * 100
        
        print(f"  Data-Driven v2: {var_v2:.2f}% energy variation")
        
        if var_v2 < 5.0:
            print(f"\n✅ Data-Driven v2 achieves excellent energy conservation!")
        elif var_v2 < 10.0:
            print(f"\n⚠️  Data-Driven v2 shows acceptable energy conservation")

if __name__ == "__main__":
    compare_data_driven_models()
    compare_data_driven_models()