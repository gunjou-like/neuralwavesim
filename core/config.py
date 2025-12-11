from dataclasses import dataclass
from typing import Optional
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
    """初期条件の定義"""
    wave_type: str = "gaussian"
    center: float = 5.0
    width: float = 1.0
    height: float = 1.0
    data: Optional[np.ndarray] = None
    
    def generate(self, x: np.ndarray) -> np.ndarray:
        """
        初期波形を生成
        
        Args:
            x: 空間座標配列
        
        Returns:
            u0: 初期変位 (nx,)
        """
        if self.data is not None:
            # カスタムデータが指定されている場合
            if len(self.data) != len(x):
                raise ValueError(
                    f"カスタムデータのサイズが不一致です\n"
                    f"  期待: {len(x)}, 実際: {len(self.data)}"
                )
            return self.data.copy()
        
        elif self.wave_type == "gaussian":
            # ガウシアンパルス
            return self.height * np.exp(-((x - self.center)**2) / (2 * self.width**2))
        
        elif self.wave_type == "sine":
            # 正弦波
            k = 2 * np.pi / self.width  # 波数
            return self.height * np.sin(k * x)
        
        elif self.wave_type == "custom":
            # カスタムジェネレータ関数がある場合
            if hasattr(self, '_custom_generator'):
                return self._custom_generator(x)
            else:
                raise ValueError("カスタム波形にはdataまたは_custom_generatorが必要です")
        
        else:
            raise ValueError(f"未対応の波形タイプ: {self.wave_type}")
    
    def validate(self, dx: float, L: float):
        """
        初期条件の妥当性を検証
        
        Args:
            dx: 空間刻み
            L: 領域長
        
        Raises:
            ValueError: 初期条件が不適切な場合
        """
        # カスタムデータの場合は波形タイプの検証をスキップ
        if self.wave_type not in ["gaussian", "sine", "custom"]:
            raise ValueError(
                f"波形タイプが不正です: {self.wave_type}\n"
                f"使用可能: gaussian, sine, custom"
            )
        
        # 境界条件のチェック（境界から十分離れているか）
        margin = max(3 * self.width, 2 * dx)  # 3σ or 2グリッド分のマージン
        
        min_center = margin
        max_center = L - margin
        
        if not (min_center <= self.center <= max_center):
            raise ValueError(
                f"初期波形の中心が境界に近すぎます\n"
                f"  現在: center = {self.center:.2f}\n"
                f"  推奨範囲: {min_center:.2f} ≤ center ≤ {max_center:.2f}\n"
                f"  (境界から {margin:.2f} 以上離す)"
            )
        
        # 振幅のチェック
        if self.height <= 0:
            raise ValueError(f"振幅は正の値である必要があります: {self.height}")
        
        # 幅のチェック
        if self.width <= 0:
            raise ValueError(f"幅は正の値である必要があります: {self.width}")
        
        # 幅が大きすぎないかチェック
        if self.width > L / 4:
            raise ValueError(
                f"波形の幅が大きすぎます\n"
                f"  現在: width = {self.width:.2f}\n"
                f"  推奨: width ≤ {L/4:.2f} (領域長の1/4以下)"
            )