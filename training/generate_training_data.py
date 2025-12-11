"""
Generate Training Data for Data-Driven Model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from core.config import PhysicsParams, InitialCondition
from models.factory import ModelFactory


def generate_diverse_initial_conditions(n_samples=100, L=10.0, min_margin=2.0):
    """
    Generate diverse initial conditions
    
    Args:
        n_samples: Number of samples
        L: Domain length
        min_margin: Minimum margin from boundaries
    
    Returns:
        list of InitialCondition objects
    """
    initial_conditions = []
    
    # Safe range for center position
    center_min = min_margin
    center_max = L - min_margin
    
    # 1. Gaussian pulses (60% of data)
    for _ in range(int(n_samples * 0.6)):
        width = np.random.uniform(0.5, 1.5)
        # Ensure center is safe considering width
        safe_margin = max(3 * width, min_margin)
        safe_center_min = safe_margin
        safe_center_max = L - safe_margin
        
        ic = InitialCondition(
            wave_type="gaussian",
            center=np.random.uniform(safe_center_min, safe_center_max),
            width=width,
            height=np.random.uniform(0.5, 1.5)
        )
        initial_conditions.append(ic)
    
    # 2. Sine waves (30% of data)
    for _ in range(int(n_samples * 0.3)):
        width = np.random.uniform(0.5, 1.5)
        
        ic = InitialCondition(
            wave_type="sine",
            center=L / 2,  # Center for sine wave
            width=width,   # Controls wavelength
            height=np.random.uniform(0.5, 1.5)
        )
        initial_conditions.append(ic)
    
    # 3. Multi-peak Gaussians (10% of data)
    for _ in range(int(n_samples * 0.1)):
        width = np.random.uniform(0.5, 1.0)
        safe_margin = max(3 * width, min_margin)
        
        # Two peaks well-separated
        center1 = np.random.uniform(safe_margin, L/2 - width)
        center2 = np.random.uniform(L/2 + width, L - safe_margin)
        height = np.random.uniform(0.5, 1.0)
        
        def multi_peak_generator(x):
            peak1 = height * np.exp(-((x - center1)**2) / (2 * width**2))
            peak2 = height * np.exp(-((x - center2)**2) / (2 * width**2))
            return peak1 + peak2
        
        ic = InitialCondition(
            wave_type="custom",
            center=L / 2,  # Not used for custom, but set to middle for validation
            width=width,
            height=height,
            data=None
        )
        ic._custom_generator = multi_peak_generator
        initial_conditions.append(ic)
    
    return initial_conditions


def augment_wave_data(wave_history, noise_level=0.01, n_augmentations=2):
    """
    Data augmentation for wave histories
    
    Args:
        wave_history: Original wave history (nt, nx)
        noise_level: Noise standard deviation
        n_augmentations: Number of augmented copies
    
    Returns:
        list of augmented wave histories
    """
    augmented_data = [wave_history]  # Include original
    
    for _ in range(n_augmentations):
        # 1. Add Gaussian noise
        noise = np.random.normal(0, noise_level, wave_history.shape)
        noisy = wave_history + noise
        augmented_data.append(noisy)
        
        # 2. Amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        scaled = wave_history * scale
        augmented_data.append(scaled)
    
    return augmented_data


def generate_training_data(
    n_samples=50,
    use_augmentation=True,
    n_augmentations=2,
    noise_level=0.01,
    save_path='training/data/wave_training_data.npz'
):
    """
    Generate training dataset
    
    Args:
        n_samples: Number of base samples
        use_augmentation: Whether to use data augmentation
        n_augmentations: Number of augmentations per sample
        noise_level: Noise level for augmentation
        save_path: Path to save dataset
    """
    print("=" * 70)
    print("Training Data Generation")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Base samples: {n_samples}")
    print(f"  Augmentation: {use_augmentation}")
    if use_augmentation:
        print(f"  Augmentations per sample: {n_augmentations}")
        print(f"  Noise level: {noise_level}")
        total_samples = n_samples * (1 + 2 * n_augmentations)
        print(f"  Total samples (with augmentation): {total_samples}")
    
    # Physics parameters
    params = PhysicsParams(
        nx=100,
        nt=100,
        c=1.0,
        dt=0.05,
        dx=0.1,
        L=10.0,
        T_max=5.0
    )
    
    print(f"\nPhysics Parameters:")
    print(f"  Grid: {params.nx} x {params.nt}")
    print(f"  Wave speed: c = {params.c} m/s")
    print(f"  Time step: dt = {params.dt} s")
    print(f"  Spatial step: dx = {params.dx} m")
    print(f"  Domain: [0, {params.L}] m")
    print(f"  Time: [0, {params.T_max}] s")
    
    # Generate initial conditions with safe margins
    print(f"\nGenerating initial conditions...")
    initial_conditions = generate_diverse_initial_conditions(
        n_samples=n_samples,
        L=params.L,
        min_margin=2.0  # Minimum 2m from boundaries
    )
    
    # Distribution statistics
    gaussian_count = sum(1 for ic in initial_conditions if ic.wave_type == "gaussian")
    sine_count = sum(1 for ic in initial_conditions if ic.wave_type == "sine")
    custom_count = sum(1 for ic in initial_conditions if ic.wave_type == "custom")
    
    print(f"  Gaussian pulses: {gaussian_count} ({gaussian_count/n_samples*100:.1f}%)")
    print(f"  Sine waves: {sine_count} ({sine_count/n_samples*100:.1f}%)")
    print(f"  Multi-peak: {custom_count} ({custom_count/n_samples*100:.1f}%)")
    
    # Physics-based model for ground truth
    physics_model = ModelFactory.create("physics")
    
    # Generate wave histories
    print(f"\nGenerating wave histories...")
    all_wave_histories = []
    
    pbar = tqdm(initial_conditions, desc="Simulating")
    for ic in pbar:
        try:
            # Handle custom generators
            if ic.wave_type == "custom" and hasattr(ic, '_custom_generator'):
                x = np.linspace(0, params.L, params.nx)
                ic.data = ic._custom_generator(x)
            
            # Generate ground truth using physics-based model
            wave_history = physics_model.predict(ic, params)
            
            # Validate energy conservation
            energies = []
            for t in range(1, params.nt - 1):
                u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * params.dt)
                u_x = np.gradient(wave_history[t], params.dx)
                
                K = 0.5 * np.sum(u_t**2) * params.dx
                P = 0.5 * params.c**2 * np.sum(u_x**2) * params.dx
                E = K + P
                energies.append(E)
            
            E_variation = (np.max(energies) - np.min(energies)) / np.mean(energies) * 100
            
            # Only include if energy is well-conserved
            if E_variation < 10.0:
                if use_augmentation:
                    augmented = augment_wave_data(
                        wave_history,
                        noise_level=noise_level,
                        n_augmentations=n_augmentations
                    )
                    all_wave_histories.extend(augmented)
                else:
                    all_wave_histories.append(wave_history)
                
                pbar.set_postfix({'Energy var': f'{E_variation:.2f}%', 'Status': '✅'})
            else:
                pbar.set_postfix({'Energy var': f'{E_variation:.2f}%', 'Status': '❌ Rejected'})
        
        except ValueError as e:
            # Skip invalid initial conditions
            pbar.set_postfix({'Status': f'⚠️ Skipped: {str(e)[:30]}...'})
            continue
    
    all_wave_histories = np.array(all_wave_histories)
    
    print(f"\n✅ Generated {len(all_wave_histories)} wave histories")
    print(f"   Shape: {all_wave_histories.shape}")
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Mean amplitude: {np.mean(np.abs(all_wave_histories)):.4f}")
    print(f"  Max amplitude: {np.max(np.abs(all_wave_histories)):.4f}")
    print(f"  Std amplitude: {np.std(all_wave_histories):.4f}")
    
    # Calculate energy statistics
    all_energies = []
    for wave_history in all_wave_histories[:10]:
        energies = []
        for t in range(1, params.nt - 1):
            u_t = (wave_history[t+1] - wave_history[t-1]) / (2 * params.dt)
            u_x = np.gradient(wave_history[t], params.dx)
            
            K = 0.5 * np.sum(u_t**2) * params.dx
            P = 0.5 * params.c**2 * np.sum(u_x**2) * params.dx
            E = K + P
            energies.append(E)
        
        E_var = (np.max(energies) - np.min(energies)) / np.mean(energies) * 100
        all_energies.append(E_var)
    
    print(f"  Energy variation (sampled): {np.mean(all_energies):.2f}% ± {np.std(all_energies):.2f}%")
    
    # Save dataset
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # ★ 辞書ではなく個別の配列として保存
    np.savez_compressed(
        save_path,
        wave_histories=all_wave_histories,
        # Physics parameters (individual arrays)
        nx=params.nx,
        nt=params.nt,
        c=params.c,
        dt=params.dt,
        dx=params.dx,
        L=params.L,
        T_max=params.T_max,
        # Generation parameters (individual values)
        n_samples=n_samples,
        use_augmentation=use_augmentation,
        n_augmentations=n_augmentations,
        noise_level=noise_level
    )
    
    print(f"\n✅ Dataset saved: {save_path}")
    print(f"   File size: {Path(save_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Visualization
    visualize_samples(all_wave_histories, params, save_path.replace('.npz', '_samples.png'))
    
    return all_wave_histories, params


def visualize_samples(wave_histories, params, save_path):
    """Visualize sample wave histories"""
    import matplotlib.pyplot as plt
    
    n_samples_to_plot = min(6, len(wave_histories))
    indices = np.random.choice(len(wave_histories), n_samples_to_plot, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    t = np.arange(params.nt) * params.dt
    x = np.linspace(0, params.L, params.nx)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        X, T = np.meshgrid(x, t)
        im = ax.pcolormesh(X, T, wave_histories[idx], cmap='RdBu_r', shading='auto')
        
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Time t (s)')
        ax.set_title(f'Sample {idx}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sample visualization saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('--samples', type=int, default=50, help='Number of base samples')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--augmentations', type=int, default=2, help='Number of augmentations per sample')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise level for augmentation')
    
    args = parser.parse_args()
    
    generate_training_data(
        n_samples=args.samples,
        use_augmentation=not args.no_augmentation,
        n_augmentations=args.augmentations,
        noise_level=args.noise
    )