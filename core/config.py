from dataclasses import dataclass
from typing import Optional

@dataclass
class PhysicsParams:
    """物理シミュレーションのパラメータ"""
    nx: int = 100          # 空間グリッド数
    nt: int = 200          # 時間ステップ数
    c: float = 1.0         # 波の速度
    dt: float = 0.05       # 時間刻み
    dx: float = 0.1        # 空間刻み
    L: float = 10.0        # 空間領域の長さ
    T_max: float = 10.0    # 最大時刻
    
    @property
    def courant_number(self) -> float:
        """CFL条件のクーラン数"""
        return self.c * self.dt / self.dx
    
    def validate(self) -> bool:
        """安定性条件のチェック"""
        if self.courant_number > 1.0:
            raise ValueError(f"CFL condition violated: C = {self.courant_number} > 1.0")
        return True

@dataclass
class InitialCondition:
    """初期条件の定義"""
    wave_type: str = "gaussian"  # gaussian, custom, zero
    data: Optional[list] = None  # custom の場合の波形データ
    
    # ガウスパルスのパラメータ
    center: float = 5.0
    width: float = 1.0
    height: float = 1.0