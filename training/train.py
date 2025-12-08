import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 物理シミュレーション (教師データ生成)
# ==========================================
class WaveSimulator:
    def __init__(self, nx=100, nt=200, c=1.0, dt=0.05, dx=0.1):
        self.nx = nx  # 空間分割数
        self.nt = nt  # 時間ステップ数
        self.c = c    # 波の速さ
        self.dt = dt
        self.dx = dx
        # クーラン数 (安定条件: C <= 1)
        self.C = c * dt / dx
        assert self.C <= 1.0, f"Unstable condition! C={self.C}"

    def step(self, u_prev, u_curr):
        """差分法による1ステップ更新 (u_nextを計算)"""
        u_next = np.zeros_like(u_curr)
        # 固定端境界条件なので、両端(0と-1)は0のまま、内側だけ計算
        # u_tt = c^2 * u_xx
        # u_next = 2*u_curr - u_prev + (C^2) * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        u_next[1:-1] = 2*u_curr[1:-1] - u_prev[1:-1] + \
                       (self.C**2) * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
        return u_next

    def generate_sample(self):
        """1つの初期条件から時系列データを生成"""
        u = np.zeros((self.nt, self.nx))
        
        # ランダムな初期条件 (ガウスパルス)
        x = np.linspace(0, 10, self.nx)
        center = np.random.uniform(2, 8)
        width = np.random.uniform(0.5, 1.5)
        height = np.random.uniform(0.5, 1.5)
        
        # t=0, t=1 (初期速度0として近似スタート)
        u[0] = height * np.exp(-((x - center)**2) / (2 * width**2))
        u[1] = u[0] # 初期速度0
        
        # シミュレーション実行
        for t in range(1, self.nt - 1):
            u[t+1] = self.step(u[t-1], u[t])
            
        return u

def create_dataset(num_samples=100):
    """学習用データセットを作成"""
    sim = WaveSimulator()
    X_data = []
    y_data = []
    
    print(f"Generating {num_samples} simulation samples...")
    for _ in range(num_samples):
        wave_history = sim.generate_sample()
        # 入力: 現在の波形 (t) と 1つ前の波形 (t-1) のセット -> 速度情報をAIに与えるため
        # 出力: 次の波形 (t+1)
        for t in range(1, sim.nt - 1):
            # Input: shape (2, nx) -> Flattenして (2*nx) にしてもよいし、2chとして扱ってもよい
            # ここではシンプルにMLPに入れるため Flatten します
            current_state = np.concatenate([wave_history[t], wave_history[t-1]])
            next_state = wave_history[t+1]
            
            X_data.append(current_state)
            y_data.append(next_state)
            
    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

# ==========================================
# 2. モデル定義 (PyTorch)
# ==========================================
try:
    # Prefer absolute import; when running this script directly the package root may not be on sys.path.
    from app.model import WavePredictor
except ImportError:
    import sys
    import os
    # Add parent directory to sys.path so 'app' can be imported when running as a script.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.model import WavePredictor

# ==========================================
# 3. 学習実行
# ==========================================
def main():
    # パラメータ
    NX = 100
    INPUT_SIZE = NX * 2  # (t) と (t-1) の2フレーム分を入力
    OUTPUT_SIZE = NX     # (t+1) を予測
    HIDDEN_SIZE = 256
    EPOCHS = 20
    BATCH_SIZE = 32

    # データ生成
    X, y = create_dataset(num_samples=50) # 50シミュレーション分 (データ数は 50 * 200step ≒ 10000)
    
    # Tensor化
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # モデル・オプティマイザ
    model = WavePredictor(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start Training...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # 保存
    torch.save(model.state_dict(), "app/wave_model.pth")
    print("Model saved to 'app/wave_model.pth'")
    
    # 学習曲線の確認（任意）
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    main()