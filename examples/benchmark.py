import torch
from fusion_kan import FusionKANLayer

# 1. Define Layer
# Input: 64, Output: 64, Grid Size: 32
layer = FusionKANLayer(64, 64, grid_size=32).cuda()

# 2. Forward Pass
x = torch.randn(10000, 64).cuda()
y = layer(x)  # [10000, 64]

# 3. Backward Pass (Fully Differentiable)
loss = y.sum()
loss.backward()