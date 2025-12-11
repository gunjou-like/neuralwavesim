import numpy as np
from .config import PhysicsParams, InitialCondition

class WaveSolver:
    """
    差分法による波動方程式ソルバー
    エネルギー保存を改善した高精度版
    """
    
    def __init__(self, params: PhysicsParams):
        self.params = params
        params.validate()  # 安定性チェック
        self.C_sq = params.courant_number ** 2
    
    def generate_initial_wave(self, ic: InitialCondition) -> np.ndarray:
        """初期波形を生成"""
        x = np.linspace(0, self.params.L, self.params.nx)
        
        if ic.wave_type == "gaussian":
            return ic.height * np.exp(-((x - ic.center)**2) / (2 * ic.width**2))
        elif ic.wave_type == "custom" and ic.data:
            return np.array(ic.data)
        elif ic.wave_type == "zero":
            return np.zeros(self.params.nx)
        else:
            raise ValueError(f"Unknown wave type: {ic.wave_type}")
    
    def step(self, u_prev: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
        """
        1ステップの時間発展
        
        Args:
            u_prev: t-1 の波形 (nx,)
            u_curr: t の波形 (nx,)
        
        Returns:
            u_next: t+1 の波形 (nx,)
        """
        u_next = np.zeros_like(u_curr)
        
        # 固定端境界条件（両端は0のまま）
        u_next[1:-1] = (
            2 * u_curr[1:-1] 
            - u_prev[1:-1]
            + self.C_sq * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
        )
        
        return u_next
    
    def solve(self, initial_condition: InitialCondition) -> np.ndarray:
        """
        時間発展全体を計算（改良版）
        
        Args:
            initial_condition: 初期条件
        
        Returns:
            u: 波動の時空間分布 (nt, nx)
        """
        u = np.zeros((self.params.nt, self.params.nx))
        
        # ステップ1: 初期変位を設定
        u[0] = self.generate_initial_wave(initial_condition)
        
        # ステップ2: u[1] を半ステップ法で計算（★改良）
        # 初期速度 v(x, t=0) = 0 を仮定すると:
        # u(t=dt) = u(0) + dt*v(0) + 0.5*dt^2*u_tt(0)
        # v(0) = 0 なので:
        # u(t=dt) = u(0) + 0.5*dt^2*c^2*u_xx(0)
        
        u_xx_0 = np.zeros(self.params.nx)
        u_xx_0[1:-1] = (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / (self.params.dx**2)
        
        u[1] = u[0] + 0.5 * (self.params.c * self.params.dt)**2 * u_xx_0
        
        # 境界条件を厳密に適用
        u[1, 0] = 0.0
        u[1, -1] = 0.0
        
        # ステップ3: t=2以降を通常の差分法で計算
        for t in range(1, self.params.nt - 1):
            u[t+1] = self.step(u[t-1], u[t])
        
        return u