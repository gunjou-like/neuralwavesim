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
    
    def __init__(self, model_path='models/checkpoints/data_driven_v2.pth'):
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
        """予測実行"""
        # 初期条件生成
        x = np.linspace(0, params.L, params.nx)
        u_current = initial_condition.generate(x)
        u_prev = u_current.copy()
        
        # ✅ デバッグ: 初期状態を確認
        print(f"\n[Data-Driven Model] Prediction Start:")
        print(f"  Initial condition: max={np.max(u_current):.4f}, min={np.min(u_current):.4f}")
        
        # 履歴保存
        wave_history = [u_current.copy()]
        
        # 時間積分
        self.model.eval()
        with torch.no_grad():
            for t in range(1, params.nt):
                # モデル入力
                u_tensor = torch.tensor(u_current, dtype=torch.float32).unsqueeze(0)
                
                # 予測
                u_next = self.model(u_tensor).squeeze(0).numpy()
                
                # ✅ デバッグ: 各ステップで異常値をチェック
                if t % 50 == 0 or np.max(np.abs(u_next)) > 10.0:
                    print(f"  Step {t}/{params.nt}: max={np.max(u_next):.4f}, min={np.min(u_next):.4f}, mean={np.mean(u_next):.4f}")
                    if np.max(np.abs(u_next)) > 10.0:
                        print(f"    ⚠️ Warning: Large amplitude detected!")
                
                # 履歴保存
                wave_history.append(u_next.copy())
                
                # 更新
                u_prev = u_current.copy()
                u_current = u_next.copy()
        
        result = np.array(wave_history)
        print(f"[Data-Driven Model] Prediction Complete: shape={result.shape}")
        
        return result
    
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