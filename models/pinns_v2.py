"""
Improved PINNs Model for Inference
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from models.base import WaveModel
from core.config import PhysicsParams, InitialCondition

class PINNsModel_v2(WaveModel):
    """Improved Physics-Informed Neural Network Model"""
    
    def __init__(self, model_path='models/pinns_v2.pth'):
        """
        Initialize improved PINNs model
        
        Args:
            model_path: Path to trained model
        """
        from training.train_pinns_v2 import WavePINNs_v2
        
        self.model = WavePINNs_v2(layers=[2, 50, 50, 50, 50, 1])
        self._model_path = model_path
        
        # Load trained weights
        if Path(model_path).exists():
            # ★ weights_only=False を追加（信頼できるソースからのモデル）
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Loaded PINNs v2 model from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print(f"   Using untrained model")
    
    @property
    def model_type(self) -> str:
        """Return model type identifier"""
        return "pinns-v2"
    
    def predict(self, initial_condition: InitialCondition, params: PhysicsParams) -> np.ndarray:
        """
        Predict wave evolution
        
        Args:
            initial_condition: Initial condition
            params: Physics parameters
        
        Returns:
            wave_history: (nt, nx) array
        """
        x_grid = np.linspace(0, params.L, params.nx)
        t_grid = np.linspace(0, params.T_max, params.nt)
        
        wave_history = np.zeros((params.nt, params.nx))
        
        with torch.no_grad():
            for i, t in enumerate(t_grid):
                x_torch = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32)
                t_torch = torch.tensor([[t]], dtype=torch.float32).repeat(params.nx, 1)
                
                u_pred = self.model(x_torch, t_torch)
                wave_history[i, :] = u_pred.numpy().flatten()
        
        return wave_history
    
    def predict_next_step(self, current_state: np.ndarray, params: PhysicsParams) -> np.ndarray:
        """
        Predict next time step (required by base class but not used in PINNs)
        
        Args:
            current_state: Current wave state (nx,)
            params: Physics parameters
        
        Returns:
            next_state: Next wave state (nx,)
        """
        # PINNs doesn't use step-by-step prediction
        # This is a placeholder implementation
        raise NotImplementedError(
            "PINNs v2 uses continuous time prediction via predict() method. "
            "Use predict() instead of predict_next_step()."
        )