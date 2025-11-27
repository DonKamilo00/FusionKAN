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
        
        # Grid configuration
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        
        self.is_output = is_output
        
        # Weights: [Out, In, Coeffs]
        num_coeffs = grid_size + spline_order
        
        # [FIX] Initialize with scale_noise directly (0.1), do NOT divide by grid_size
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, num_coeffs) * scale_noise)
        
        # Base Linear Weights
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * (1 / math.sqrt(in_features)))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable Scales (Per Output Channel)
        # This is critical for convergence on high-frequency functions
        self.scale_base = nn.Parameter(torch.ones(out_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * scale_spline)
        
        # Activation & Norm
        if not is_output:
            self.layer_norm = nn.LayerNorm(out_features)
            self.prelu = nn.PReLU()
            
        self.base_activation = base_activation()

    def forward(self, x):
        # x shape: [Batch, In]
        
        # 1. Base Linear Path
        base_output = F.linear(self.base_activation(x), self.base_weight, self.base_bias)
        # Apply learnable scale (broadcast over batch)
        base_output = base_output * self.scale_base.view(1, -1)
        
        # 2. Spline Path (Fused CUDA)
        if x.device.type == 'cuda':
            spline_output = FusionKANFunction.apply(
                x, 
                self.spline_weight, 
                self.grid_size, 
                self.grid_min, 
                self.grid_max
            )
        else:
            raise NotImplementedError("FusionKAN only supports CUDA tensors for now.")
            
        # Apply learnable scale to spline output
        spline_output = spline_output * self.scale_spline.view(1, -1)
        
        # 3. Combine
        y = base_output + spline_output
        
        if self.is_output:
            return y
        
        return self.prelu(self.layer_norm(y))