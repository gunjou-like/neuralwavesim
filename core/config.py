from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class PhysicsParams:
    """物理パラメータ"""
    nx: int = 100
    nt: int = 200
    c: float = 1.0
    dt: float = 0.05
    dx: float = 0.1
    L: float = 10.0
    T_max: float = 10.0
    
    @property
    def courant_number(self) -> float:
        """CFL数（Courant-Friedrichs-Lewy条件）"""
        return self.c * self.dt / self.dx
    
    def validate(self):
        """安定性条件のチェック"""
        # CFL条件
        if self.courant_number > 1.0:
            raise ValueError(
                f"CFL条件違反: C={self.courant_number:.3f} > 1.0\n"
                f"dt を {self.dx / self.c:.4f} 以下にしてください"
            )
        
        # 空間解像度の警告
        if self.dx > self.L / 50:
            print(f"⚠️  空間解像度が低い: dx={self.dx:.4f}, L={self.L:.2f}")
            print(f"   推奨: dx ≤ {self.L / 100:.4f}")
        
        # 時間ステップ数の警告
        if self.nt < 100:
            print(f"⚠️  時間ステップ数が少ない: nt={self.nt}")
        
        # CFL数の推奨値
        if self.courant_number < 0.1:
            print(f"⚠️  CFL数が小さすぎます: C={self.courant_number:.3f}")
            print(f"   エネルギー保存性が悪化する可能性があります")

@dataclass
class InitialCondition:
    """初期条件の設定"""
    wave_type: str = "gaussian"
    center: float = 5.0
    width: float = 1.0
    height: float = 1.0
    data: Optional[List[float]] = None
    
    def validate(self, dx: float, L: float):
        """
        初期条件の妥当性チェック
        
        Args:
            dx: 空間刻み
            L: 領域の長さ
        
        Raises:
            ValueError: 初期条件が不適切な場合
        """
        if self.wave_type == "gaussian":
            # ★ 幅の検証を厳格化（数値分散を考慮）
            # 特性波長 λ_c = 2π·σ
            # 高周波成分まで考慮: λ_min ≈ 2π·σ/3
            # 必要解像度: Δx ≤ λ_min/5 (精度確保)
            
            characteristic_wavelength = 2 * np.pi * self.width
            required_points = 10  # 1波長あたり10点以上
            
            min_width = required_points * dx / (2 * np.pi)
            
            if self.width < min_width:
                raise ValueError(
                    f"ガウスパルスの幅が狭すぎます\n"
                    f"  現在: width = {self.width:.4f}\n"
                    f"  最小: width ≥ {min_width:.4f}\n"
                    f"  (特性波長を {required_points} 点以上で解像するため)\n"
                    f"  推奨: width ≥ {min_width * 1.5:.4f}"
                )
            
            # 警告レベル（推奨値）
            recommended_width = min_width * 1.5
            if self.width < recommended_width:
                import warnings
                warnings.warn(
                    f"パルス幅が推奨値より小さい: width = {self.width:.4f}\n"
                    f"推奨: width ≥ {recommended_width:.4f}\n"
                    f"数値分散により精度が低下する可能性があります",
                    UserWarning
                )
            
            # 幅の検証（広すぎないか）
            max_width = L / 4
            if self.width > max_width:
                raise ValueError(
                    f"ガウスパルスの幅が広すぎます\n"
                    f"  現在: width = {self.width:.4f}\n"
                    f"  最大: width ≤ {max_width:.4f} (L/4)\n"
                    f"  領域長: L = {L:.2f}"
                )
            
            # 中心位置の検証
            margin = max(3 * self.width, 5 * dx)  # ★ より厳格に
            min_center = margin
            max_center = L - margin
            
            if self.center < min_center or self.center > max_center:
                raise ValueError(
                    f"初期波形の中心が境界に近すぎます\n"
                    f"  現在: center = {self.center:.2f}\n"
                    f"  推奨範囲: {min_center:.2f} ≤ center ≤ {max_center:.2f}\n"
                    f"  (境界から {margin:.2f} 以上離す)"
                )
            
            # 振幅の検証（警告のみ）
            if self.height > 10.0:
                import warnings
                warnings.warn(
                    f"振幅が大きい: height = {self.height:.2f}\n"
                    f"数値誤差が増加する可能性があります"
                )
        
        elif self.wave_type == "custom":
            if self.data is None:
                raise ValueError("wave_type='custom' の場合、data を指定してください")
            
            expected_length = int(L / dx) + 1
            if len(self.data) != expected_length:
                raise ValueError(
                    f"data の長さが不正です\n"
                    f"  期待: {expected_length}\n"
                    f"  実際: {len(self.data)}"
                )