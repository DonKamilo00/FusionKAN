import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .functional import FusionKANFunction

class FusionKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                 base_activation=nn.SiLU, grid_eps=0.02, 
                 grid_range=[-3, 3], # Expanded range for stability
                 is_output=False,
                 use_node_activation=False): # NEW: Toggle for "Pure" KAN mode
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

        # Weights: [Out, In, Coeffs]
        num_coeffs = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_coeffs))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # --- INITIALIZATION ---
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        
        self.scale_base = nn.Parameter(torch.ones(out_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * scale_spline)
        
        # Auto-Scaling (Critical for fixed CUDA grid)
        self.input_norm = nn.BatchNorm1d(in_features)
        
        # Optional Node Activation (LayerNorm + PReLU)
        # We initialize these even if unused to avoid errors if user toggles flag later,
        # but in forward pass they are skipped.
        self.layer_norm = nn.LayerNorm(out_features)
        self.prelu = nn.PReLU()
            
        self.base_activation = base_activation()

    def forward(self, x):
        # 1. Auto-Scaling
        x_norm = self.input_norm(x)
        
        # 2. Base Linear Path
        base_output = F.linear(self.base_activation(x_norm), self.base_weight, self.base_bias)
        base_output = base_output * self.scale_base.view(1, -1)
        
        # 3. Spline Path (Fused CUDA Kernel)
        # This ALWAYS runs the optimized C++ code
        if x.device.type == 'cuda':
            spline_output = FusionKANFunction.apply(
                x_norm, 
                self.spline_weight, 
                self.grid_size, 
                self.grid_min, 
                self.grid_max
            )
        else:
            raise NotImplementedError("FusionKAN only supports CUDA tensors.")
            
        spline_output = spline_output * self.scale_spline.view(1, -1)
        
        # 4. Combine
        y = base_output + spline_output
        
        # 5. Output / Activation Logic
        # If it's an output layer OR pure mode is requested, return raw sum
        if self.is_output or not self.use_node_activation:
            return y
        
        # Otherwise, apply modern deep learning activations
        return self.prelu(self.layer_norm(y))