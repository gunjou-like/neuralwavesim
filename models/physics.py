import numpy as np
from .base import WaveModel
from core.config import PhysicsParams, InitialCondition
from core.solver import WaveSolver

class PhysicsBasedModel(WaveModel):
    """
    差分法による物理ベースモデル
    PINNs/データ駆動型と同じインターフェースでアクセス可能
    """
    
    def __init__(self):
        self.solver = None
    
    def predict(
        self, 
        initial_condition: InitialCondition,
        params: PhysicsParams
    ) -> np.ndarray:
        """差分法でシミュレーション実行"""
        self.solver = WaveSolver(params)
        return self.solver.solve(initial_condition)
    
    def predict_next_step(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> np.ndarray:
        """1ステップ予測（差分法）"""
        if self.solver is None:
            # デフォルトパラメータで初期化
            params = PhysicsParams(nx=len(current))
            self.solver = WaveSolver(params)
        
        return self.solver.step(previous, current)
    
    @property
    def model_type(self) -> str:
        return "physics"