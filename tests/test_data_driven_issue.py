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
    
    # ✅ 修正: UIと同じパラメータを明示的に設定
    params = PhysicsParams(
        nx=100,
        nt=10,      # 短縮版 (200だと時間がかかるため)
        c=1.0,
        dt=0.05,    # ✅ 追加
        dx=0.1,     # ✅ 追加 (L=10, nx=100 → dx=0.1)
        L=10.0,     # ✅ 追加
        T_max=0.5   # ✅ 追加 (10 steps × 0.05 = 0.5s)
    )
    
    ic = InitialCondition(
        wave_type="gaussian", 
        center=5.0, 
        width=1.0, 
        height=1.0
    )
    
    # ✅ デバッグ: パラメータを表示
    print("=" * 70)
    print("Test Configuration:")
    print("=" * 70)
    print(f"Physics Parameters:")
    print(f"  nx={params.nx}, nt={params.nt}, c={params.c}")
    print(f"  dt={params.dt}, dx={params.dx}, L={params.L}, T_max={params.T_max}")
    print(f"\nInitial Condition:")
    print(f"  Type: {ic.wave_type}")
    print(f"  Center: {ic.center}, Width: {ic.width}, Height: {ic.height}")
    
    # ✅ 初期波形を事前に確認
    x = np.linspace(0, params.L, params.nx)
    initial_wave = ic.generate(x)
    print(f"\nGenerated Initial Wave:")
    print(f"  Max: {np.max(initial_wave):.4f}")
    print(f"  Min: {np.min(initial_wave):.4f}")
    print(f"  Mean: {np.mean(initial_wave):.4f}")
    print("=" * 70)
    
    # シミュレーション実行
    print("\nRunning simulation...")
    result = model.predict(ic, params)
    
    print("\n" + "=" * 70)
    print("Simulation Results:")
    print("=" * 70)
    print(f"Output shape: {result.shape}")
    print(f"t=0  (Initial): max={np.max(result[0]):.4f}, min={np.min(result[0]):.4f}")
    print(f"t=1  (Step 1):  max={np.max(result[1]):.4f}, min={np.min(result[1]):.4f}")
    print(f"t=2  (Step 2):  max={np.max(result[2]):.4f}, min={np.min(result[2]):.4f}")
    print(f"t=5  (Step 5):  max={np.max(result[5]):.4f}, min={np.min(result[5]):.4f}")
    print(f"t=9  (Final):   max={np.max(result[9]):.4f}, min={np.min(result[9]):.4f}")
    print(f"Overall: max={np.max(result):.4f}, min={np.min(result):.4f}")
    
    # ✅ 初期条件との比較
    print(f"\n⚠️  Initial Condition Preserved?")
    print(f"  Expected (IC): max={np.max(initial_wave):.4f}")
    print(f"  Actual (t=0):  max={np.max(result[0]):.4f}")
    print(f"  Difference:    {np.abs(np.max(initial_wave) - np.max(result[0])):.6f}")
    
    if not np.allclose(result[0], initial_wave, atol=1e-6):
        print("  ❌ Initial condition NOT preserved!")
    else:
        print("  ✅ Initial condition preserved")
    
    print("=" * 70)
    
    # 初期波形のプロット
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    x = np.linspace(0, params.L, params.nx)
    
    # 上段: 初期数ステップ
    for i, ax in enumerate(axes[0]):
        ax.plot(x, result[i], linewidth=2)
        ax.set_title(f"t={i} (dt={i*params.dt:.2f}s)", fontweight='bold')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 下段: 中間〜最終ステップ
    steps = [3, 5, 9]
    for i, (step, ax) in enumerate(zip(steps, axes[1])):
        ax.plot(x, result[step], linewidth=2, color='C3')
        ax.set_title(f"t={step} (dt={step*params.dt:.2f}s)", fontweight='bold')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    save_path = Path("tests/results/data_driven_issue.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved: {save_path}")
    plt.show()
    
    return result


if __name__ == "__main__":
    result = test_initial_condition()