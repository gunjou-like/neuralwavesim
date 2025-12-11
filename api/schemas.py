from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class InitialWaveRequest(BaseModel):
    """初期波形の定義"""
    wave_type: Literal["gaussian", "custom", "zero"] = "gaussian"
    data: Optional[List[float]] = None  # custom の場合のみ必須
    
    # ガウスパルスのパラメータ
    center: float = Field(default=5.0, ge=0, le=10)
    width: float = Field(default=1.0, gt=0)
    height: float = Field(default=1.0, gt=0)

class SimulationRequest(BaseModel):
    """シミュレーションリクエスト（統一インターフェース）"""
    model_type: Literal["physics", "data-driven", "pinns"] = "physics"
    
    # 物理パラメータ
    nx: int = Field(default=100, ge=10, le=500)
    nt: int = Field(default=200, ge=10, le=1000)
    c: float = Field(default=1.0, gt=0)
    dt: float = Field(default=0.05, gt=0)
    dx: float = Field(default=0.1, gt=0)
    L: float = Field(default=10.0, gt=0)
    T_max: float = Field(default=10.0, gt=0)
    
    # 初期条件
    initial_condition: InitialWaveRequest = Field(default_factory=InitialWaveRequest)

class SimulationResponse(BaseModel):
    """シミュレーション結果"""
    wave_history: List[List[float]]  # shape: (nt, nx)
    model_type: str
    params: dict
    shape: tuple
    computation_time_ms: float

class NextStepRequest(BaseModel):
    """1ステップ予測リクエスト（既存API互換）"""
    model_type: Literal["physics", "data-driven", "pinns"] = "data-driven"
    current_wave: List[float] = Field(..., min_length=100, max_length=100)
    previous_wave: List[float] = Field(..., min_length=100, max_length=100)

class NextStepResponse(BaseModel):
    """1ステップ予測結果"""
    next_wave: List[float]
    model_type: str

class HealthResponse(BaseModel):
    """ヘルスチェック"""
    status: str
    available_models: List[str]
    models_loaded: dict