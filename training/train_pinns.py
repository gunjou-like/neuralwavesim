# training/train_pinns.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from app.pinns_model import WavePINNs

# ハイパーパラメータ
EPOCHS = 5000
LR = 1e-3
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0

# 物理パラメータ
C = 1.0
L = 10.0
T_MAX = 10.0

def generate_training_data(n_pde=1000, n_bc=100, n_ic=100):
    """PINNs用のトレーニングポイント生成"""
    # PDE残差計算用 (requires_grad=True が重要)
    x_pde = torch.rand(n_pde, 1, requires_grad=True) * L
    t_pde = torch.rand(n_pde, 1, requires_grad=True) * T_MAX
    
    # 境界条件: x=0, x=L で u=0
    t_bc = torch.rand(n_bc, 1) * T_MAX
    x_bc_left = torch.zeros(n_bc // 2, 1)
    x_bc_right = torch.ones(n_bc // 2, 1) * L
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    t_bc = torch.cat([t_bc[:n_bc//2], t_bc[n_bc//2:]], dim=0)
    
    # 初期条件: t=0 でガウスパルス
    x_ic = torch.linspace(0, L, n_ic).unsqueeze(1)
    t_ic = torch.zeros(n_ic, 1)
    u_ic = torch.exp(-((x_ic - L/2)**2) / 2.0)
    
    return x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic

def train():
    """PINNs モデルの学習"""
    print("Starting PINNs training...")
    
    model = WavePINNs()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    losses = []
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # ★ 修正: 毎エポックごとにトレーニングデータを再生成
        x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic = generate_training_data()
        
        # 1. PDE残差損失
        residual = model.pde_residual(x_pde, t_pde, c=C)
        loss_pde = torch.mean(residual**2)
        
        # 2. 境界条件損失
        u_bc = model(x_bc, t_bc)
        loss_bc = torch.mean(u_bc**2)
        
        # 3. 初期条件損失
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic)**2)
        
        # 総損失
        loss = loss_pde * LAMBDA_PDE + loss_bc * LAMBDA_BC + loss_ic
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f} "
                  f"(PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, IC: {loss_ic.item():.6f})")
    
    # モデル保存
    save_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'wave_pinns.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # 学習曲線プロット
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINNs Training Loss')
    plt.grid(True)
    plt.savefig('pinns_training_loss.png')
    print("Training curve saved to pinns_training_loss.png")
    plt.show()

if __name__ == "__main__":
    train()