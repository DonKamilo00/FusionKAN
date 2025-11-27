import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .functional import FusionKANFunction

class FusionKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1], is_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.is_output = is_output
        
        num_coeffs = grid_size + spline_order
        
        # --- WEIGHTS ---
        # [Out, In, Coeffs]
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_coeffs))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # --- INITIALIZATION (Matching Original KAN) ---
        # 1. Splines: Uniform[-scale, scale]
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)
        
        # 2. Base: Kaiming Uniform (He Init)
        # We use the same scaling factor as the paper (sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        
        # --- SCALES ---
        self.scale_base = nn.Parameter(torch.ones(out_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * scale_spline)
        
        if not is_output:
            self.layer_norm = nn.LayerNorm(out_features)
            self.prelu = nn.PReLU()
            
        self.base_activation = base_activation()

    def forward(self, x):
        # 1. Base Path
        # Apply base weights
        base = F.linear(self.base_activation(x), self.base_weight, self.base_bias)
        # Apply learnable scale
        base = base * self.scale_base.view(1, -1)
        
        # 2. Spline Path (Fused CUDA)
        if x.device.type == 'cuda':
            spline = FusionKANFunction.apply(
                x, 
                self.spline_weight, 
                self.grid_size, 
                self.grid_min, 
                self.grid_max
            )
        else:
            raise NotImplementedError("FusionKAN only supports CUDA tensors.")
            
        # Apply learnable scale
        spline = spline * self.scale_spline.view(1, -1)
        
        # 3. Combine
        y = base + spline
        
        if self.is_output:
            return y
        
        return self.prelu(self.layer_norm(y))