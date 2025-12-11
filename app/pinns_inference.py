import torch
import numpy as np
from pinns_model import WavePINNs

class PINNsInference:
    """PINNs モデルによる推論クラス"""
    
    def __init__(self, model_path="wave_pinns.pth"):
        self.model = WavePINNs()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
    
    def predict_wave_evolution(self, nx=100, nt=200, L=10.0, T_MAX=10.0):
        """
        時空間全体の波動を予測
        
        Args:
            nx: 空間グリッド数
            nt: 時間ステップ数
            L: 空間領域の長さ
            T_MAX: 最大時刻
        
        Returns:
            u: 波動の時空間分布 (nt, nx)
        """
        x_grid = np.linspace(0, L, nx)
        t_grid = np.linspace(0, T_MAX, nt)
        
        u = np.zeros((nt, nx))
        
        with torch.no_grad():
            for i, t_val in enumerate(t_grid):
                x_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)
                t_tensor = torch.full((nx, 1), t_val, dtype=torch.float32)
                
                u_pred = self.model(x_tensor, t_tensor)
                u[i, :] = u_pred.squeeze().numpy()
        
        return u
    
    def predict_next_step(self, current_wave, previous_wave, dt=0.05, dx=0.1):
        """
        データ駆動型モデルとの互換性のため、1ステップ予測も提供
        
        Args:
            current_wave: 現在の波形 (nx,)
            previous_wave: 1つ前の波形 (nx,) ← 使わないが互換性のため
            dt: 時間刻み
            dx: 空間刻み
        
        Returns:
            next_wave: 次ステップの波形 (nx,)
        """
        # PINNs は時刻を直接指定できるので、dt 後の波形を予測
        nx = len(current_wave)
        x = np.linspace(0, 10.0, nx)
        t_next = dt  # 仮に次のステップを dt 後とする
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        t_tensor = torch.full((nx, 1), t_next, dtype=torch.float32)
        
        with torch.no_grad():
            next_wave = self.model(x_tensor, t_tensor).squeeze().numpy()
        
        return next_wave