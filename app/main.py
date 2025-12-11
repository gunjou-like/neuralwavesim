# app/main.py (修正版)
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import os
from model import WavePredictor
from pinns_model import WavePINNs
from pinns_inference import PINNsInference

app = FastAPI()

# --- 既存モデル (データ駆動型) ---
model_data_driven = WavePredictor(200, 256, 100)
if os.path.exists("wave_model.pth"):
    model_data_driven.load_state_dict(torch.load("wave_model.pth", map_location='cpu'))
    model_data_driven.eval()

# --- PINNs モデル ---
pinns_inference = None
if os.path.exists("wave_pinns.pth"):
    pinns_inference = PINNsInference("wave_pinns.pth")
    print("✅ PINNs model loaded successfully")

class WaveInput(BaseModel):
    wave_data: list[float]
    use_pinns: bool = False

class SimulationRequest(BaseModel):
    nx: int = 100
    nt: int = 200
    use_pinns: bool = True

@app.post("/predict")
def predict(input_data: WaveInput):
    """1ステップ予測（既存互換）"""
    if input_data.use_pinns and pinns_inference:
        # PINNs による予測
        current = np.array(input_data.wave_data[:100])
        previous = np.array(input_data.wave_data[100:]) if len(input_data.wave_data) > 100 else current
        
        next_wave = pinns_inference.predict_next_step(current, previous)
        return {"next_wave": next_wave.tolist(), "method": "PINNs"}
    else:
        # データ駆動型予測
        data = np.array(input_data.wave_data, dtype=np.float32)
        tensor_in = torch.from_numpy(data).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model_data_driven(tensor_in)
        
        return {"next_wave": prediction.squeeze().tolist(), "method": "Data-Driven"}

@app.post("/simulate")
def simulate(request: SimulationRequest):
    """時空間全体のシミュレーション（PINNs専用）"""
    if not request.use_pinns or not pinns_inference:
        return {"error": "PINNs model not available"}
    
    wave_history = pinns_inference.predict_wave_evolution(
        nx=request.nx,
        nt=request.nt
    )
    
    return {
        "wave_history": wave_history.tolist(),
        "shape": wave_history.shape,
        "method": "PINNs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "data_driven_loaded": model_data_driven is not None,
        "pinns_loaded": pinns_inference is not None
    }