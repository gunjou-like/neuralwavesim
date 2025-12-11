import numpy as np
from .config import PhysicsParams, InitialCondition

class WaveSolver:
    """
    差分法による波動方程式ソルバー
    境界条件処理を改善した安定版
    """
    
    def __init__(self, params: PhysicsParams):
        self.params = params
        params.validate()
        self.C_sq = params.courant_number ** 2
    
    def generate_initial_wave(self, ic: InitialCondition) -> np.ndarray:
        """初期波形を生成（境界チェック付き）"""
        x = np.linspace(0, self.params.L, self.params.nx)
        
        if ic.wave_type == "gaussian":
            wave = ic.height * np.exp(-((x - ic.center)**2) / (2 * ic.width**2))
            
            # ★ 境界条件を厳密に適用
            wave[0] = 0.0
            wave[-1] = 0.0
            
            # ★ 境界付近の値も減衰させる（スムーズな境界処理）
            # 境界から 3dx 以内を減衰
            boundary_width = 3
            for i in range(boundary_width):
                # 左境界
                factor = i / boundary_width
                wave[i] *= factor
                # 右境界
                wave[-(i+1)] *= factor
            
            return wave
        
        elif ic.wave_type == "custom" and ic.data:
            wave = np.array(ic.data)
            wave[0] = 0.0
            wave[-1] = 0.0
            return wave
        
        elif ic.wave_type == "zero":
            return np.zeros(self.params.nx)
        
        else:
            raise ValueError(f"Unknown wave type: {ic.wave_type}")
    
    def step(self, u_prev: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
        """1ステップの時間発展"""
        u_next = np.zeros_like(u_curr)
        
        # 内部点の更新
        u_next[1:-1] = (
            2 * u_curr[1:-1] 
            - u_prev[1:-1]
            + self.C_sq * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
        )
        
        # 境界条件（固定端）
        u_next[0] = 0.0
        u_next[-1] = 0.0
        
        return u_next
    
    def solve(self, initial_condition: InitialCondition) -> np.ndarray:
        """
        時間発展全体を計算（改良版）
        
        Args:
            initial_condition: 初期条件
        
        Returns:
            u: 波動の時空間分布 (nt, nx)
        """
        # 初期条件の妥当性チェック
        initial_condition.validate(self.params.dx, self.params.L)
        
        u = np.zeros((self.params.nt, self.params.nx))
        
        # ステップ1: 初期変位を設定
        u[0] = self.generate_initial_wave(initial_condition)
        
        # ステップ2: u[1] を半ステップ法で計算
        u_xx_0 = np.zeros(self.params.nx)
        u_xx_0[1:-1] = (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / (self.params.dx**2)
        
        # 境界では2階微分もゼロ
        u_xx_0[0] = 0.0
        u_xx_0[-1] = 0.0
        
        u[1] = u[0] + 0.5 * (self.params.c * self.params.dt)**2 * u_xx_0
        
        # 境界条件を厳密に適用
        u[1, 0] = 0.0
        u[1, -1] = 0.0
        
        # ステップ3: t=2以降を通常の差分法で計算
        for t in range(1, self.params.nt - 1):
            u[t+1] = self.step(u[t-1], u[t])
            
            # ★ 数値安定性チェック
            max_val = np.max(np.abs(u[t+1]))
            if np.isnan(max_val) or np.isinf(max_val):
                raise RuntimeError(
                    f"数値発散を検出しました\n"
                    f"  時刻: t = {t * self.params.dt:.3f}\n"
                    f"  ステップ: {t}/{self.params.nt}\n"
                    f"  最大値: {max_val}"
                )
            
            # 過度な増幅の警告
            initial_max = np.max(np.abs(u[0]))
            if max_val > 100 * initial_max:
                import warnings
                warnings.warn(
                    f"数値的な増幅を検出: max(|u|) = {max_val:.2e} "
                    f"(初期値の {max_val/initial_max:.1f} 倍)"
                )
        
        return u