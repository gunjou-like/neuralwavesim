from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
import time
import numpy as np

from core.solver import WaveSolver
from core.config import PhysicsParams, InitialCondition
from models.factory import ModelFactory

app = FastAPI(
    title="Neural Wave Simulator API",
    description="3つのモデル（物理ベース/データ駆動型/PINNs）で波動シミュレーション",
    version="2.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストスキーマ
class InitialConditionRequest(BaseModel):
    wave_type: str = "gaussian"
    center: float = 5.0
    width: float = 1.0
    height: float = 1.0
    data: Optional[List[float]] = None

class SimulationRequest(BaseModel):
    model_type: Literal["physics", "data-driven-v2", "pinns", "pinns-v2"]  # ★ 修正
    nx: int = Field(100, ge=50, le=500)
    nt: int = Field(200, ge=50, le=1000)
    c: float = Field(1.0, gt=0, le=5.0)
    initial_condition: InitialConditionRequest
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ["physics", "data-driven-v2", "pinns", "pinns-v2"]
        if v not in allowed:
            raise ValueError(f"model_type は {allowed} のいずれかを指定してください")
        return v

@app.post("/simulate")
def simulate(request: SimulationRequest):
    """シミュレーション実行"""
    start_time = time.time()
    
    try:
        # パラメータ構築
        L = 10.0
        T_max = request.nt * 0.05
        
        params = PhysicsParams(
            nx=request.nx,
            nt=request.nt,
            c=request.c,
            dt=0.05,
            dx=L / request.nx,
            L=L,
            T_max=T_max
        )
        
        # ✅ デバッグログ追加
        print(f"\n{'='*70}")
        print(f"Simulation Request:")
        print(f"  Model: {request.model_type}")
        print(f"  nx={params.nx}, nt={params.nt}, c={params.c}")
        print(f"  dt={params.dt}, dx={params.dx:.4f}")
        print(f"  L={params.L}, T_max={params.T_max}")
        print(f"{'='*70}")
        
        # 初期条件
        initial_condition = InitialCondition(
            wave_type=request.initial_condition.wave_type,
            center=request.initial_condition.center,
            width=request.initial_condition.width,
            height=request.initial_condition.height,
            data=request.initial_condition.data
        )
        
        # ✅ 初期波形の確認
        x_grid = np.linspace(0, params.L, params.nx)
        initial_wave = initial_condition.generate(x_grid)
        
        print(f"\nInitial Condition:")
        print(f"  Type: {initial_condition.wave_type}")
        print(f"  Center: {initial_condition.center}, Width: {initial_condition.width}, Height: {initial_condition.height}")
        print(f"  Generated wave: max={np.max(initial_wave):.4f}, min={np.min(initial_wave):.4f}, mean={np.mean(initial_wave):.4f}")
        print(f"  Grid points: {len(x_grid)}")
        
        # モデル実行
        model = ModelFactory.create(request.model_type)
        
        # ✅ モデル情報
        print(f"\nModel Info:")
        print(f"  Type: {type(model).__name__}")
        if hasattr(model, 'model'):
            print(f"  Model loaded: {model.model is not None}")
            if hasattr(model.model, 'eval'):
                print(f"  Model in eval mode: {not model.model.training}")
        
        wave_history = model.predict(initial_condition, params)
        
        # ✅ 結果の確認
        print(f"\nSimulation Results:")
        print(f"  Output shape: {wave_history.shape}")
        print(f"  Initial (t=0): max={np.max(wave_history[0]):.4f}, min={np.min(wave_history[0]):.4f}")
        print(f"  Final (t={params.nt-1}): max={np.max(wave_history[-1]):.4f}, min={np.min(wave_history[-1]):.4f}")
        print(f"  Overall: max={np.max(wave_history):.4f}, min={np.min(wave_history):.4f}")
        
        # 計算時間
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "model_type": request.model_type,
            "wave_history": wave_history.tolist(),
            "params": {
                "nx": params.nx,
                "nt": params.nt,
                "c": params.c,
                "dt": params.dt,
                "dx": params.dx,
                "L": params.L,
                "T_max": params.T_max
            },
            "computation_time_ms": computation_time
        }
    
    except HTTPException:
        # 既に処理済みのエラーは再送出
        raise
    
    except ValueError as e:
        # 検証エラー
        raise HTTPException(status_code=400, detail=str(e))
    
    except RuntimeError as e:
        # 数値発散など
        raise HTTPException(status_code=500, detail=f"数値計算エラー: {str(e)}")
    
    except Exception as e:
        # ★ デバッグ情報を追加
        import traceback
        error_detail = f"内部エラー: {str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/")
def root():
    return {
        "message": "Neural Wave Simulator API",
        "version": "2.0",
        "available_models": [
            "physics",
            "data-driven-v2",
            "pinns",
            "pinns-v2"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "service": "NeuralWaveSim API",
        "version": "2.1"
    }