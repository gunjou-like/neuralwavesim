import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from models.data_driven import DataDrivenModel
from core.config import PhysicsParams, InitialCondition

def test_initial_condition():
    """初期条件が保持されるか確認"""
    
    # モデル作成
    model = DataDrivenModel()
    
    # パラメータ設定
    params = PhysicsParams(nx=100, nt=10)
    ic = InitialCondition(wave_type="gaussian", center=5.0, width=1.0, height=1.0)
    
    # シミュレーション実行
    result = model.predict(ic, params)
    
    print("=" * 50)
    print("データ駆動型モデルのテスト")
    print("=" * 50)
    print(f"t=0 の最大値: {np.max(result[0]):.4f} (期待値: ~1.0)")
    print(f"t=0 の最小値: {np.min(result[0]):.4f}")
    print(f"t=0 がゼロか: {np.allclose(result[0], 0)}")
    print(f"t=1 の最大値: {np.max(result[1]):.4f}")
    print(f"t=2 の最大値: {np.max(result[2]):.4f}")
    print("=" * 50)
    
    # 初期波形のプロット
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(result[0])
    axes[0].set_title("t=0 (初期条件)")
    axes[0].set_ylim([-0.1, 1.5])
    axes[0].grid(True)
    
    axes[1].plot(result[1])
    axes[1].set_title("t=1")
    axes[1].set_ylim([-0.1, 1.5])
    axes[1].grid(True)
    
    axes[2].plot(result[2])
    axes[2].set_title("t=2 (NN予測)")
    axes[2].set_ylim([-0.1, 1.5])
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("data_driven_issue.png")
    print("図を保存: data_driven_issue.png")
    plt.show()

if __name__ == "__main__":
    test_initial_condition()