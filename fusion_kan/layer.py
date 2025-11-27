import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .functional import FusionKANFunction

class FusionKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                 base_activation=nn.SiLU, grid_eps=0.02, 
                 grid_range=[-2, 2], # <--- OPTIMIZED: Covers 95% of Normal Dist
                 is_output=False,
                 use_node_activation=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.scale_spline = scale_spline
        self.is_output = is_output
        self.use_node_activation = use_node_activation

        # Weights
        num_coeffs = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_coeffs))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # Init
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        self.scale_base = nn.Parameter(torch.ones(out_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * scale_spline)
        
        # --- CRITICAL FIX: Affine=False ---
        # We don't want BatchNorm learning a shift/scale while Splines also learn a shift/scale.
        # This locks the input to N(0,1) so the splines have a stationary target.
        self.input_norm = nn.BatchNorm1d(in_features, affine=False)
        
        self.layer_norm = nn.LayerNorm(out_features)
        self.prelu = nn.PReLU()
        self.base_activation = base_activation()

    def forward(self, x):
        # 1. Norm
        x_norm = self.input_norm(x)
        
        # 2. Base
        base_output = F.linear(self.base_activation(x_norm), self.base_weight, self.base_bias)
        base_output = base_output * self.scale_base.view(1, -1)
        
        # 3. Spline
        if x.device.type == 'cuda':
            spline_output = FusionKANFunction.apply(
                x_norm, 
                self.spline_weight, 
                self.grid_size, 
                self.grid_min, 
                self.grid_max
            )
        else:
            raise NotImplementedError("FusionKAN only supports CUDA.")
            
        spline_output = spline_output * self.scale_spline.view(1, -1)
        
        # 4. Sum
        y = base_output + spline_output
        
        if self.is_output or not self.use_node_activation:
            return y
        
        return self.prelu(self.layer_norm(y))