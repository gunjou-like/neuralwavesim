import torch
import numpy as np
from pathlib import Path
from .base import WaveModel
from core.config import PhysicsParams, InitialCondition

class WavePredictor(torch.nn.Module):
    """既存のデータ駆動型NN（変更なし）"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)


class DataDrivenModel(WaveModel):
    """
    データ駆動型モデルのラッパー
    統一インターフェースに適合
    """
    
    def __init__(self, model_path: str = "models/checkpoints/wave_model.pth"):
        self.device = torch.device("cpu")
        self.model = WavePredictor(200, 256, 100)
        
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
        """逐次予測で時系列を生成"""
        u = np.zeros((params.nt, params.nx))
        
        # 初期条件設定
        from core.solver import WaveSolver
        solver = WaveSolver(params)
        u[0] = solver.generate_initial_wave(initial_condition)
        u[1] = u[0]
        
        # NN による逐次予測
        for t in range(1, params.nt - 1):
            u[t+1] = self.predict_next_step(u[t], u[t-1])
        
        return u
    
    def predict_next_step(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> np.ndarray:
        """1ステップ予測"""
        input_data = np.concatenate([current, previous]).astype(np.float32)
        tensor_in = torch.from_numpy(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(tensor_in)
        
        return prediction.squeeze().cpu().numpy()
    
    @property
    def model_type(self) -> str:
        return "data-driven"