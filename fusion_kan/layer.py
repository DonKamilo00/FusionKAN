import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .functional import FusionKANFunction

class FusionKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, 
                 base_activation=nn.SiLU, grid_eps=0.02, 
                 grid_range=[-2, 2], 
                 is_output=False,
                 use_node_activation=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Learnable Grid Bounds
        # Initialize as Parameters so the optimizer can adjust them
        self.grid_min = nn.Parameter(torch.tensor(float(grid_range[0])))
        self.grid_max = nn.Parameter(torch.tensor(float(grid_range[1])))
        
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
        
        # Normalization
        # affine=False to ensure inputs are roughly N(0,1), enabling initial grid usage.
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
            # Pass Parameters directly. Autograd handles gradients for grid_min/max.
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

    @torch.no_grad()
    def upscale_grid(self, new_grid_size):
        """
        Transfers weights from the current grid size to a larger grid size 
        via linear interpolation. Crucial for Coarse-to-Fine training.
        """
        if new_grid_size <= self.grid_size:
            print(f"Skipping upscale: {new_grid_size} <= {self.grid_size}")
            return

        # Weights shape: [Out, In, Coeffs]
        # Coeffs = grid_size + spline_order
        old_coeffs = self.spline_weight.data
        out_dim, in_dim, old_num = old_coeffs.shape
        
        # Calculate new number of coefficients
        new_num = new_grid_size + self.spline_order
        
        # We treat the coefficients as a 1D signal and interpolate them.
        # Reshape to [1, Out*In, Old_Coeffs] for F.interpolate (expects [N, C, L])
        flat_coeffs = old_coeffs.view(1, out_dim * in_dim, old_num)
        
        # Interpolate
        # mode='linear' approximates B-spline subdivision fairly well for training
        new_flat_coeffs = F.interpolate(
            flat_coeffs, 
            size=new_num, 
            mode='linear', 
            align_corners=True
        )
        
        # Reshape back to [Out, In, New_Coeffs]
        new_coeffs = new_flat_coeffs.view(out_dim, in_dim, new_num)
        
        # Update Parameter
        self.spline_weight = nn.Parameter(new_coeffs)
        self.grid_size = new_grid_size
        
        print(f"Layer upscaled: Grid {old_num-self.spline_order} -> {new_grid_size}")