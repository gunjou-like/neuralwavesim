import torch
import torch.nn as nn

class WavePINNs(nn.Module):
    """
    Physics-Informed Neural Networks for Wave Equation
    
    入力: [x, t] (座標と時刻)
    出力: u(x, t) (その点での変位)
    """
    def __init__(self, layers=[2, 50, 50, 50, 1]):
        super().__init__()
        
        self.activation = nn.Tanh()
        
        # ネットワーク構築
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        """
        Args:
            x: 空間座標 (batch_size, 1)
            t: 時刻 (batch_size, 1)
        Returns:
            u: 変位 (batch_size, 1)
        """
        inputs = torch.cat([x, t], dim=1)
        

        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        u = self.layers[-1](inputs) # 最終層は活性化関数なし
        return u
    
    def pde_residual(self, x, t, c=1.0):
        """
        波動方程式の残差を計算: u_tt - c^2 * u_xx = 0
        
        Args:
            x: 空間座標 (requires_grad=True)
            t: 時刻 (requires_grad=True)
            c: 波の速度
        
        Returns:
            residual: PDE残差
        """
        u = self.forward(x, t)

        # 1階微分
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 2階微分
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # 波動方程式の残差
        residual = u_tt - c**2 * u_xx
        return residual