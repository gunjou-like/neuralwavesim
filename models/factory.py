from typing import Union
from .base import WaveModel
from .physics import PhysicsBasedModel
from .data_driven import DataDrivenModel
from .pinns import PINNsModel

class ModelFactory:
    """
    モデル生成のファクトリクラス
    エッジ・WASM移植時にここだけ変更すればよい
    """
    
    _models = {
        "physics": PhysicsBasedModel,
        "data-driven": DataDrivenModel,
        "pinns": PINNsModel,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> WaveModel:
        """
        モデルインスタンスを生成
        
        Args:
            model_type: "physics" | "data-driven" | "pinns"
            **kwargs: モデル固有のパラメータ
        
        Returns:
            WaveModel インスタンス
        
        Example:
            >>> model = ModelFactory.create("pinns", model_path="custom.pth")
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._models.keys())}"
            )
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def available_models(cls) -> list:
        """利用可能なモデル一覧"""
        return list(cls._models.keys())