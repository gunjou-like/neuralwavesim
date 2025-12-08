# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import os
from model import WavePredictor  # 同階層のmodel.pyからクラスを読み込み

app = FastAPI()

# --- 設定 ---
NX = 100
INPUT_SIZE = NX * 2
HIDDEN_SIZE = 256
OUTPUT_SIZE = NX
MODEL_PATH = "wave_model.pth"

# --- モデルのロード (起動時に1回だけ実行) ---
print("Loading model...")
model = WavePredictor(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# CPUで動かす設定 (map_location='cpu' はクラウドデプロイ時のエラー回避にも重要)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # 推論モードへ
    print("Model loaded successfully.")
else:
    print("Error: wave_model.pth not found!")

# --- データ形式の定義 ---
class WaveInput(BaseModel):
    # フロントエンドから送られてくる配列 (長さ200: 現在の波 + 1つ前の波)
    wave_data: list[float]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Neural Wave Simulator API is running."}

@app.post("/predict")
def predict(input_data: WaveInput):
    """
    現在の波形データを受け取り、次の時刻の波形を予測して返す
    """
    # 1. リストをnumpy配列へ
    data = np.array(input_data.wave_data, dtype=np.float32)
    
    # 2. PyTorchのテンソルへ変換 & バッチ次元を追加 (shape: [1, 200])
    tensor_in = torch.from_numpy(data).unsqueeze(0)
    
    # 3. AI予測実行
    with torch.no_grad():
        prediction = model(tensor_in)
    
    # 4. 結果をリストに戻してJSONで返す
    return {"next_wave": prediction.squeeze().tolist()}