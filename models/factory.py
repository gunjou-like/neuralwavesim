from typing import Union
from .base import WaveModel
from .physics import PhysicsBasedModel
from .data_driven import DataDrivenModel
from .data_driven_v2 import DataDrivenModel_v2
from .pinns import PINNsModel
from .pinns_v2 import PINNsModel_v2

class ModelFactory:
    """
    モデル生成のファクトリクラス
    エッジ・WASM移植時にここだけ変更すればよい
    """
    
    _models = {
        "physics": PhysicsBasedModel,
        "data-driven": DataDrivenModel,
        "data-driven-v2": DataDrivenModel_v2,  # ★ 追加
        "pinns": PINNsModel,
        "pinns-v2": PINNsModel_v2,  # ★ 追加
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> WaveModel:
        """
        モデルインスタンスを生成
        
        Args:
            model_type: "physics" | "data-driven" | "data-driven-v2" | "pinns" | "pinns-v2"
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