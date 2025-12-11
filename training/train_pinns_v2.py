"""
Improved PINNs Training Script with Energy Conservation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class WavePINNs_v2(nn.Module):
    """Physics-Informed Neural Network with Energy Conservation"""
    
    def __init__(self, layers=[2, 50, 50, 50, 50, 1]):
        super().__init__()
        
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = nn.Tanh()
        
        # Xavier initialization
        for layer in self.layers_list:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x, t):
        """
        Forward pass
        
        Args:
            x: Spatial coordinate (batch, 1)
            t: Time coordinate (batch, 1)
        
        Returns:
            u: Displacement (batch, 1)
        """
        inputs = torch.cat([x, t], dim=1)
        
        out = inputs
        for i, layer in enumerate(self.layers_list[:-1]):
            out = self.activation(layer(out))
        
        out = self.layers_list[-1](out)
        return out
    
    def pde_residual(self, x, t, c):
        """
        PDE Residual: ∂²u/∂t² - c² ∂²u/∂x²
        
        Args:
            x: Spatial points
            t: Time points
            c: Wave speed
        
        Returns:
            residual: PDE residual
        """
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        
        u = self.forward(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivatives
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # PDE residual
        residual = u_tt - c**2 * u_xx
        
        return residual
    
    def energy_conservation_loss(self, x, t, c, dx):
        """
        Energy Conservation Loss
        
        Total Energy: E = 1/2 ∫ [(∂u/∂t)² + c²(∂u/∂x)²] dx
        Conservation: dE/dt ≈ 0
        
        Args:
            x: Spatial grid (N, 1)
            t: Time point (1,) - same for all x
            c: Wave speed
            dx: Spatial step
        
        Returns:
            loss: Energy conservation loss
        """
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        
        # Expand t to match x
        t_expanded = t.repeat(x.shape[0], 1)
        
        u = self.forward(x, t_expanded)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u, t_expanded,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Spatial derivative
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Energy density
        kinetic_density = 0.5 * u_t**2
        potential_density = 0.5 * c**2 * u_x**2
        energy_density = kinetic_density + potential_density
        
        # Total energy (numerical integration)
        total_energy = torch.sum(energy_density) * dx
        
        return total_energy

def train_pinns_v2(
    n_epochs=10000,
    lr=1e-3,
    n_collocation=2000,
    n_boundary=200,
    n_initial=200,
    n_energy_check=10,
    save_path='models/pinns_v2.pth'
):
    """
    Train improved PINNs with energy conservation
    
    Loss Function:
        L = λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic + λ_energy * L_energy
    """
    print("=" * 70)
    print("Improved PINNs Training with Energy Conservation")
    print("=" * 70)
    
    # Parameters
    L = 10.0
    T_max = 10.0
    c = 1.0
    dx = L / 100
    
    # Loss weights
    lambda_pde = 1.0
    lambda_bc = 1.0
    lambda_ic = 1.0
    lambda_energy = 0.5
    
    print(f"\nPhysics Parameters:")
    print(f"  Domain length: L = {L} m")
    print(f"  Time domain: T_max = {T_max} s")
    print(f"  Wave speed: c = {c} m/s")
    print(f"  Spatial step: dx = {dx:.3f} m")
    
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Boundary points: {n_boundary}")
    print(f"  Initial condition points: {n_initial}")
    
    print(f"\nLoss Weights:")
    print(f"  λ_pde = {lambda_pde}")
    print(f"  λ_bc = {lambda_bc}")
    print(f"  λ_ic = {lambda_ic}")
    print(f"  λ_energy = {lambda_energy} ★ NEW")
    
    # Model
    model = WavePINNs_v2(layers=[2, 50, 50, 50, 50, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    
    # Training history
    history = {
        'loss': [],
        'loss_pde': [],
        'loss_bc': [],
        'loss_ic': [],
        'loss_energy': [],
        'energy_values': []
    }
    
    # Training loop
    print(f"\nTraining started...")
    pbar = tqdm(range(n_epochs), desc="Training")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # 1. PDE Residual Loss (Interior points)
        x_pde = torch.rand(n_collocation, 1) * L
        t_pde = torch.rand(n_collocation, 1) * T_max
        
        residual = model.pde_residual(x_pde, t_pde, c)
        loss_pde = torch.mean(residual**2)
        
        # 2. Boundary Condition Loss
        t_bc = torch.rand(n_boundary, 1) * T_max
        x_bc_left = torch.zeros(n_boundary, 1)
        x_bc_right = torch.ones(n_boundary, 1) * L
        
        u_bc_left = model(x_bc_left, t_bc)
        u_bc_right = model(x_bc_right, t_bc)
        
        loss_bc = torch.mean(u_bc_left**2) + torch.mean(u_bc_right**2)
        
        # 3. Initial Condition Loss
        x_ic = torch.rand(n_initial, 1) * L
        t_ic = torch.zeros(n_initial, 1)
        
        # Gaussian pulse: u(x, 0) = exp(-(x-5)²/(2*1²))
        u_ic_true = torch.exp(-((x_ic - 5.0)**2) / (2 * 1.0**2))
        u_ic_pred = model(x_ic, t_ic)
        
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
        
        # 4. ★ Energy Conservation Loss
        # Sample time points
        t_energy_samples = torch.linspace(0, T_max, n_energy_check).reshape(-1, 1)
        x_energy_grid = torch.linspace(0, L, 100).reshape(-1, 1)
        
        energies = []
        for t_val in t_energy_samples:
            E = model.energy_conservation_loss(x_energy_grid, t_val, c, dx)
            energies.append(E)
        
        energies_tensor = torch.stack(energies)
        
        # Energy should be constant over time
        E_mean = torch.mean(energies_tensor)
        loss_energy = torch.mean((energies_tensor - E_mean)**2)
        
        # Total Loss
        loss = (lambda_pde * loss_pde + 
                lambda_bc * loss_bc + 
                lambda_ic * loss_ic + 
                lambda_energy * loss_energy)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record history
        history['loss'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_energy'].append(loss_energy.item())
        history['energy_values'].append(energies_tensor.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4e}',
            'PDE': f'{loss_pde.item():.4e}',
            'Energy': f'{loss_energy.item():.4e}'
        })
        
        # Logging
        if (epoch + 1) % 1000 == 0:
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            print(f"  Loss PDE:    {loss_pde.item():.6e}")
            print(f"  Loss BC:     {loss_bc.item():.6e}")
            print(f"  Loss IC:     {loss_ic.item():.6e}")
            print(f"  Loss Energy: {loss_energy.item():.6e} ★")
            print(f"  Total Loss:  {loss.item():.6e}")
            print(f"  Mean Energy: {E_mean.item():.6f}")
    
    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'params': {
            'L': L,
            'T_max': T_max,
            'c': c,
            'dx': dx,
            'n_epochs': n_epochs,
            'lr': lr
        }
    }, save_path)
    
    print(f"\n✅ Model saved: {save_path}")
    
    # Plot training history
    plot_training_history(history, save_path.replace('.pth', '_history.png'))
    
    return model, history

def plot_training_history(history, save_path):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Total loss
    ax = axes[0, 0]
    ax.semilogy(epochs, history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Individual losses
    ax = axes[0, 1]
    ax.semilogy(epochs, history['loss_pde'], label='PDE', linewidth=2)
    ax.semilogy(epochs, history['loss_bc'], label='BC', linewidth=2)
    ax.semilogy(epochs, history['loss_ic'], label='IC', linewidth=2)
    ax.semilogy(epochs, history['loss_energy'], label='Energy ★', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Individual Losses', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy conservation
    ax = axes[1, 0]
    energy_over_time = np.array(history['energy_values'])  # (epochs, n_energy_check)
    
    # Plot last epoch's energy vs time
    if len(energy_over_time) > 0:
        last_energies = energy_over_time[-1]
        t_points = np.linspace(0, 10.0, len(last_energies))
        ax.plot(t_points, last_energies, 'go-', linewidth=2, markersize=6)
        ax.axhline(np.mean(last_energies), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(last_energies):.4f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Energy')
        ax.set_title('Energy vs Time (Final Epoch)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Energy variation over epochs
    ax = axes[1, 1]
    energy_stds = [np.std(e) for e in energy_over_time]
    ax.semilogy(epochs, energy_stds, 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy Std Dev')
    ax.set_title('Energy Conservation Quality', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    model, history = train_pinns_v2(
        n_epochs=10000,
        lr=1e-3,
        save_path='models/pinns_v2.pth'
    )