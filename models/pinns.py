import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .base import WaveModel
from core.config import PhysicsParams, InitialCondition

class WavePINNs(nn.Module):
    """PINNs ネットワーク（既存コードから移植）"""
    def __init__(self, layers=[2, 50, 50, 50, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        u = self.layers[-1](inputs)
        return u
    
    def pde_residual(self, x, t, c=1.0):
        """波動方程式の残差計算"""
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        return u_tt - c**2 * u_xx


class PINNsModel(WaveModel):
    """
    PINNs モデルのラッパー
    統一インターフェースに適合
    """
    
    def __init__(self, model_path: str = "models/checkpoints/wave_pinns.pth"):
        self.device = torch.device("cpu")
        self.model = WavePINNs()
        
        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.eval()
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    def predict(
        self, 
        initial_condition: InitialCondition,
        params: PhysicsParams
    ) -> np.ndarray:
        """時空間全体を一度に予測"""
        x_grid = np.linspace(0, params.L, params.nx)
        t_grid = np.linspace(0, params.T_max, params.nt)
        
        u = np.zeros((params.nt, params.nx))
        
        with torch.no_grad():
            for i, t_val in enumerate(t_grid):
                x_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)
                t_tensor = torch.full((params.nx, 1), t_val, dtype=torch.float32)
                
                u_pred = self.model(x_tensor, t_tensor)
                u[i, :] = u_pred.squeeze().numpy()
        
        return u
    
    def predict_next_step(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> np.ndarray:
        """1ステップ予測（互換性のため）"""
        # PINNs は時刻を直接指定できるので簡易実装
        nx = len(current)
        x = np.linspace(0, 10.0, nx)
        t_next = 0.05  # 仮の時間刻み
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        t_tensor = torch.full((nx, 1), t_next, dtype=torch.float32)
        
        with torch.no_grad():
            next_wave = self.model(x_tensor, t_tensor).squeeze().numpy()
        
        return next_wave
    
    @property
    def model_type(self) -> str:
        return "pinns"