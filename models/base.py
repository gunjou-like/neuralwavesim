from abc import ABC, abstractmethod
import numpy as np
from core.config import PhysicsParams, InitialCondition

class WaveModel(ABC):
    """
    全てのモデルの共通インターフェース
    エッジ・WASM移植時の互換性を保証
    """
    
    @abstractmethod
    def predict(
        self, 
        initial_condition: InitialCondition,
        params: PhysicsParams
    ) -> np.ndarray:
        """
        波動シミュレーションを実行
        
        Args:
            initial_condition: 初期波形
            params: 物理パラメータ
        
        Returns:
            wave_history: (nt, nx) の時系列データ
        """
        pass
    
    @abstractmethod
    def predict_next_step(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> np.ndarray:
        """
        1ステップの予測（既存API互換性のため）
        
        Args:
            current: 現在の波形 (nx,)
            previous: 1つ前の波形 (nx,)
        
        Returns:
            next_wave: 次ステップの波形 (nx,)
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """モデルの種類を返す"""
        pass