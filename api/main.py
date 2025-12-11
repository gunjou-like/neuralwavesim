from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from models.factory import ModelFactory
from core.config import PhysicsParams, InitialCondition
from api.schemas import (
    SimulationRequest,
    SimulationResponse,
    NextStepRequest,
    NextStepResponse,
    HealthResponse
)

app = FastAPI(
    title="Neural Wave Simulator API",
    description="統一インターフェースによる波動シミュレーション",
    version="2.0.0"
)

# CORS設定（Streamlit連携用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルにモデルをキャッシュ
_model_cache = {}

def get_model(model_type: str):
    """モデルのキャッシュ付き取得"""
    if model_type not in _model_cache:
        try:
            _model_cache[model_type] = ModelFactory.create(model_type)
            print(f"✅ {model_type} model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load {model_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    return _model_cache[model_type]


# ★ 追加: 起動時イベント（事前ロード）
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時にモデルを事前ロード"""
    print("=" * 50)
    print("Loading models...")
    print("=" * 50)
    
    for model_type in ModelFactory.available_models():
        try:
            get_model(model_type)
        except Exception as e:
            print(f"❌ Failed to load {model_type}: {e}")
    
    print("=" * 50)
    print(f"✅ {len(_model_cache)} models ready")
    print("=" * 50)


@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest):
    """
    波動シミュレーション実行（統一インターフェース）
    
    3種類のモデル（physics/data-driven/pinns）を同じAPIで利用可能
    """
    start_time = time.time()
    
    try:
        # モデル取得
        model = get_model(request.model_type)
        
        # パラメータ変換
        params = PhysicsParams(
            nx=request.nx,
            nt=request.nt,
            c=request.c,
            dt=request.dt,
            dx=request.dx,
            L=request.L,
            T_max=request.T_max
        )
        
        # 初期条件変換
        ic_data = request.initial_condition
        initial_condition = InitialCondition(
            wave_type=ic_data.wave_type,
            data=ic_data.data,
            center=ic_data.center,
            width=ic_data.width,
            height=ic_data.height
        )
        
        # シミュレーション実行
        wave_history = model.predict(initial_condition, params)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SimulationResponse(
            wave_history=wave_history.tolist(),
            model_type=model.model_type,
            params=params.__dict__,
            shape=wave_history.shape,
            computation_time_ms=round(elapsed_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-next", response_model=NextStepResponse)
def predict_next_step(request: NextStepRequest):
    """
    1ステップ予測（既存API互換）
    """
    try:
        model = get_model(request.model_type)
        
        current = np.array(request.current_wave)
        previous = np.array(request.previous_wave)
        
        next_wave = model.predict_next_step(current, previous)
        
        return NextStepResponse(
            next_wave=next_wave.tolist(),
            model_type=model.model_type
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    ヘルスチェック & 利用可能なモデル一覧
    """
    available = ModelFactory.available_models()
    
    models_loaded = {}
    for model_type in available:
        try:
            get_model(model_type)
            models_loaded[model_type] = True
        except:
            models_loaded[model_type] = False
    
    return HealthResponse(
        status="healthy",
        available_models=available,
        models_loaded=models_loaded
    )


@app.get("/")
def root():
    return {
        "message": "Neural Wave Simulator API v2.0",
        "docs": "/docs",
        "endpoints": {
            "simulate": "POST /simulate",
            "predict_next": "POST /predict-next",
            "health": "GET /health"
        }
    }