"""
Improved Data-Driven Model for Inference
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from models.base import WaveModel
from core.config import PhysicsParams, InitialCondition

class DataDrivenModel_v2(WaveModel):
    """Improved Data-Driven Wave Prediction Model"""
    
    def __init__(self, model_path='models/data_driven_v2.pth'):
        """
        Initialize improved data-driven model
        
        Args:
            model_path: Path to trained model
        """
        from training.train_data_driven_v2 import WavePredictor_v2
        
        # Default architecture
        self.model = WavePredictor_v2(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            dropout=0.1
        )
        
        self._model_path = model_path
        
        # Load trained weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Loaded Data-Driven v2 model from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print(f"   Using untrained model")
    
    @property
    def model_type(self) -> str:
        """Return model type identifier"""
        return "data-driven-v2"
    
    def predict(self, initial_condition: InitialCondition, params: PhysicsParams) -> np.ndarray:
        """
        Predict wave evolution
        
        Args:
            initial_condition: Initial condition
            params: Physics parameters
        
        Returns:
            wave_history: (nt, nx) array
        """
        # Initialize
        x = np.linspace(0, params.L, params.nx)
        current_state = initial_condition.generate(x)
        
        wave_history = np.zeros((params.nt, params.nx))
        wave_history[0] = current_state
        
        # Iterative prediction
        with torch.no_grad():
            for t in range(1, params.nt):
                state_tensor = torch.tensor(
                    current_state.reshape(1, -1),
                    dtype=torch.float32
                )
                
                next_state = self.model(state_tensor)
                current_state = next_state.numpy().flatten()
                
                wave_history[t] = current_state
        
        return wave_history
    
    def predict_next_step(self, current_state: np.ndarray, params: PhysicsParams) -> np.ndarray:
        """
        Predict next time step
        
        Args:
            current_state: Current wave state (nx,)
            params: Physics parameters
        
        Returns:
            next_state: Next wave state (nx,)
        """
        with torch.no_grad():
            state_tensor = torch.tensor(
                current_state.reshape(1, -1),
                dtype=torch.float32
            )
            
            next_state = self.model(state_tensor)
            
        return next_state.numpy().flatten()