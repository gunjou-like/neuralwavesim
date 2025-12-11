import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Core モジュールから物理シミュレーターをインポート
from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition

# モデル定義
class WavePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

def generate_training_data(num_samples=1000, nx=100, nt=200):
    """物理シミュレーションで学習データ生成"""
    print("Generating training data...")
    
    X_train = []
    y_train = []
    
    params = PhysicsParams(nx=nx, nt=nt)
    
    for i in range(num_samples):
        # ランダムな初期条件
        center = np.random.uniform(2.0, 8.0)
        width = np.random.uniform(0.5, 2.0)
        height = np.random.uniform(0.5, 1.5)
        
        ic = InitialCondition(
            wave_type="gaussian",
            center=center,
            width=width,
            height=height
        )
        
        # 物理シミュレーション実行
        solver = WaveSolver(params)
        wave_history = solver.solve(ic)
        
        # データペア作成: (t-1, t) → t+1
        for t in range(1, nt - 1):
            X_train.append(np.concatenate([wave_history[t], wave_history[t-1]]))
            y_train.append(wave_history[t+1])
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_samples} samples...")
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train

def train_model(epochs=1000, batch_size=64, lr=1e-3):
    """モデル学習"""
    print("=" * 50)
    print("Training Data-Driven Wave Predictor")
    print("=" * 50)
    
    # データ生成
    X_train, y_train = generate_training_data(num_samples=500)
    
    # モデル・オプティマイザ
    model = WavePredictor(200, 256, 100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 学習ループ
    losses = []
    
    for epoch in range(epochs):
        # ランダムサンプリング
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = torch.from_numpy(X_train[indices])
        y_batch = torch.from_numpy(y_train[indices])
        
        # 順伝播
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # モデル保存（★ 修正: 正しいパスに保存）
    save_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "wave_model.pth"
    
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    # 学習曲線プロット
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Data-Driven Model Training Loss')
    plt.grid(True)
    plt.savefig('data_driven_training_loss.png')
    print("✅ Training curve saved to data_driven_training_loss.png")
    plt.show()

if __name__ == "__main__":
    train_model(epochs=1000, batch_size=64, lr=1e-3)