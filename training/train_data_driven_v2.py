"""
Improved Data-Driven Model Training with Energy Regularization
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class WavePredictor_v2(nn.Module):
    """Improved Data-Driven Wave Predictor with Energy Conservation"""
    
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size)
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input wave state (batch, input_size)
        
        Returns:
            next_state: Predicted next state (batch, input_size)
        """
        # Reshape for LSTM: (batch, seq_len=1, input_size)
        x = x.unsqueeze(1)
        
        # LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        
        # FC layers
        out = self.dropout(self.activation(self.fc1(out)))
        out = self.fc2(out)
        
        return out


def energy_regularization_loss(pred, input_wave, c, dx):
    """
    Energy regularization loss
    
    Constraint: Energy of prediction should be close to energy of input
    
    Args:
        pred: Predicted wave (batch, nx)
        input_wave: Input wave (batch, nx)
        c: Wave speed
        dx: Spatial step
    
    Returns:
        loss: Energy regularization loss
    """
    # Input energy (potential energy from spatial derivative)
    u_x_input = torch.gradient(input_wave, spacing=(dx,), dim=1)[0]
    E_input = 0.5 * c**2 * torch.sum(u_x_input**2, dim=1) * dx
    
    # Predicted energy
    u_x_pred = torch.gradient(pred, spacing=(dx,), dim=1)[0]
    E_pred = 0.5 * c**2 * torch.sum(u_x_pred**2, dim=1) * dx
    
    # Energy conservation constraint
    return torch.mean((E_pred - E_input)**2)


def smoothness_loss(pred, dx):
    """
    Smoothness regularization loss
    
    Penalize large second derivatives (curvature)
    
    Args:
        pred: Predicted wave (batch, nx)
        dx: Spatial step
    
    Returns:
        loss: Smoothness loss
    """
    # First derivative
    u_x = torch.gradient(pred, spacing=(dx,), dim=1)[0]
    
    # Second derivative
    u_xx = torch.gradient(u_x, spacing=(dx,), dim=1)[0]
    
    # Penalize large curvature
    return torch.mean(u_xx**2)


def amplitude_constraint_loss(pred, input_wave):
    """
    Amplitude constraint loss
    
    Prevent unbounded growth
    
    Args:
        pred: Predicted wave (batch, nx)
        input_wave: Input wave (batch, nx)
    
    Returns:
        loss: Amplitude constraint loss
    """
    # Maximum amplitude should not grow significantly
    max_input = torch.max(torch.abs(input_wave), dim=1)[0]
    max_pred = torch.max(torch.abs(pred), dim=1)[0]
    
    # Penalize amplitude growth
    growth = torch.relu(max_pred - 1.2 * max_input)  # Allow 20% tolerance
    
    return torch.mean(growth**2)


def train_data_driven_v2(
    train_data_path='training/data/wave_training_data.npz',
    n_epochs=200,
    batch_size=32,
    lr=1e-3,
    save_path='models/data_driven_v2.pth'
):
    """
    Train improved data-driven model
    
    Loss Function:
        L = L_mse + λ_energy * L_energy + λ_smooth * L_smooth + λ_amp * L_amp
    """
    print("=" * 70)
    print("Improved Data-Driven Model Training")
    print("=" * 70)
    
    # Load training data
    if not Path(train_data_path).exists():
        print(f"\n❌ Training data not found: {train_data_path}")
        print("   Please run: python training/generate_training_data.py")
        return None, None
    
    data = np.load(train_data_path)
    wave_histories = data['wave_histories']
    
    # ★ 個別に読み込み
    c = float(data['c'])
    dx = float(data['dx'])
    nx = int(data['nx'])
    
    print(f"\nData loaded:")
    print(f"  Wave histories shape: {wave_histories.shape}")
    print(f"  Physics params: c={c}, dx={dx}, nx={nx}")
    
    # Create training pairs
    X_train = []
    y_train = []
    
    for wave_history in wave_histories:
        for t in range(len(wave_history) - 1):
            X_train.append(wave_history[t])
            y_train.append(wave_history[t + 1])
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    print(f"\nTraining pairs: {len(X_train)}")
    
    # Loss weights
    lambda_energy = 0.5
    lambda_smooth = 0.1
    lambda_amp = 0.3
    
    print(f"\nLoss Weights:")
    print(f"  λ_mse = 1.0")
    print(f"  λ_energy = {lambda_energy} ★")
    print(f"  λ_smooth = {lambda_smooth} ★")
    print(f"  λ_amp = {lambda_amp} ★")
    
    # Model
    model = WavePredictor_v2(
        input_size=nx,
        hidden_size=128,
        num_layers=2,
        dropout=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10  # ★ verbose を削除
    )
    
    # Training history
    history = {
        'loss': [],
        'loss_mse': [],
        'loss_energy': [],
        'loss_smooth': [],
        'loss_amp': [],
        'lr': []  # ★ 学習率の履歴を追加
    }
    
    # Dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Training loop
    print(f"\nTraining started...")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    best_loss = float('inf')
    pbar = tqdm(range(n_epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        
        epoch_loss = 0
        epoch_mse = 0
        epoch_energy = 0
        epoch_smooth = 0
        epoch_amp = 0
        
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Prediction
            pred = model(X_batch)
            
            # 1. MSE Loss
            loss_mse = nn.MSELoss()(pred, y_batch)
            
            # 2. Energy Regularization
            loss_energy = energy_regularization_loss(pred, X_batch, c, dx)
            
            # 3. Smoothness Constraint
            loss_smooth = smoothness_loss(pred, dx)
            
            # 4. Amplitude Constraint
            loss_amp = amplitude_constraint_loss(pred, X_batch)
            
            # Total Loss
            loss = (loss_mse + 
                   lambda_energy * loss_energy + 
                   lambda_smooth * loss_smooth + 
                   lambda_amp * loss_amp)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_energy += loss_energy.item()
            epoch_smooth += loss_smooth.item()
            epoch_amp += loss_amp.item()
        
        # Average losses
        n_batches = len(dataloader)
        epoch_loss /= n_batches
        epoch_mse /= n_batches
        epoch_energy /= n_batches
        epoch_smooth /= n_batches
        epoch_amp /= n_batches
        
        # Record history
        history['loss'].append(epoch_loss)
        history['loss_mse'].append(epoch_mse)
        history['loss_energy'].append(epoch_energy)
        history['loss_smooth'].append(epoch_smooth)
        history['loss_amp'].append(epoch_amp)
        history['lr'].append(optimizer.param_groups[0]['lr'])  # ★ 現在の学習率を記録
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # ★ 学習率変更時に通知
        if new_lr != old_lr:
            print(f"\n📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{epoch_loss:.4e}',
            'MSE': f'{epoch_mse:.4e}',
            'Energy': f'{epoch_energy:.4e}',
            'LR': f'{new_lr:.2e}'  # ★ 学習率を表示
        })
        
        # Logging
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            print(f"  Loss MSE:    {epoch_mse:.6e}")
            print(f"  Loss Energy: {epoch_energy:.6e} ★")
            print(f"  Loss Smooth: {epoch_smooth:.6e} ★")
            print(f"  Loss Amp:    {epoch_amp:.6e} ★")
            print(f"  Total Loss:  {epoch_loss:.6e}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'history': history,
                'params': {
                    'nx': nx,
                    'c': c,
                    'dx': dx,
                    'hidden_size': 128,
                    'num_layers': 2
                }
            }, save_path)
    
    print(f"\n✅ Training completed!")
    print(f"   Best loss: {best_loss:.6e}")
    print(f"   Model saved: {save_path}")
    
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
    
    # MSE loss
    ax = axes[0, 1]
    ax.semilogy(epochs, history['loss_mse'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Prediction MSE', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Energy & Smoothness
    ax = axes[1, 0]
    ax.semilogy(epochs, history['loss_energy'], label='Energy', linewidth=2)
    ax.semilogy(epochs, history['loss_smooth'], label='Smoothness', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Regularization Losses ★', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Amplitude
    ax = axes[1, 1]
    ax.semilogy(epochs, history['loss_amp'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Amplitude Loss')
    ax.set_title('Amplitude Constraint ★', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    model, history = train_data_driven_v2(
        n_epochs=200,
        batch_size=32,
        lr=1e-3,
        save_path='models/data_driven_v2.pth'
    )