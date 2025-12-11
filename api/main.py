from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import time

from core.config import PhysicsParams, InitialCondition
from models.factory import ModelFactory

app = FastAPI(
    title="Neural Wave Simulator API",
    description="3つのモデル（物理ベース/データ駆動型/PINNs）で波動シミュレーション",
    version="1.0.0"
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
    model_type: str = Field(..., description="physics, data-driven, または pinns")
    nx: int = Field(100, ge=50, le=500)
    nt: int = Field(200, ge=50, le=1000)
    c: float = Field(1.0, gt=0, le=5.0)
    initial_condition: InitialConditionRequest
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ["physics", "data-driven", "pinns"]
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
        
        initial_condition = InitialCondition(
            wave_type=request.initial_condition.wave_type,
            center=request.initial_condition.center,
            width=request.initial_condition.width,
            height=request.initial_condition.height,
            data=request.initial_condition.data
        )
        
        # ★ 初期条件の検証
        try:
            initial_condition.validate(params.dx, params.L)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # モデル実行
        model = ModelFactory.create(request.model_type, params)
        wave_history = model.predict(initial_condition)
        
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
        # その他のエラー
        raise HTTPException(status_code=500, detail=f"内部エラー: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Neural Wave Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /simulate": "シミュレーション実行",
            "GET /docs": "API ドキュメント"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}