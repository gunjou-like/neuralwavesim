import argparse
import sys
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from models.factory import ModelFactory
from core.config import PhysicsParams, InitialCondition

def main():
    parser = argparse.ArgumentParser(description="Wave Simulator CLI")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["physics", "data-driven", "pinns"],
        default="physics",
        help="Model type to use"
    )
    parser.add_argument(
        "--nx", type=int, default=100, help="Spatial grid points"
    )
    parser.add_argument(
        "--nt", type=int, default=200, help="Time steps"
    )
    parser.add_argument(
        "--output", type=str, default="output.npy", help="Output file path"
    )
    
    args = parser.parse_args()
    
    # モデル作成
    model = ModelFactory.create(args.model)
    
    # パラメータ設定
    params = PhysicsParams(nx=args.nx, nt=args.nt)
    initial_condition = InitialCondition(wave_type="gaussian")
    
    # シミュレーション実行
    print(f"Running {args.model} simulation...")
    result = model.predict(initial_condition, params)
    
    # 結果保存
    np.save(args.output, result)
    print(f"✅ Saved to {args.output} (shape: {result.shape})")

if __name__ == "__main__":
    main()