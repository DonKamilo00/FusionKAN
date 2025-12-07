import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# Import your library
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion_kan.layer import FusionKANLayer

class FusionKANSDF(nn.Module):
    def __init__(self, hidden_dim=64, grid_size=32):
        super().__init__()
        # 3D Inputs (x,y,z) -> Hidden
        self.layer1 = FusionKANLayer(3, hidden_dim, grid_size=grid_size)
        self.layer2 = FusionKANLayer(hidden_dim, hidden_dim, grid_size=grid_size)
        # Hidden -> 1 Output (Distance)
        # The last layer is usually linear in SDFs, but we use KAN here too.
        # We assume the output range of KAN is sufficient.
        self.layer3 = FusionKANLayer(hidden_dim, 1, grid_size=grid_size, is_output=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def get_sphere_data(batch_size, device):
    # Sample random points in [-1, 1]
    coords = (torch.rand(batch_size, 3, device=device) * 2) - 1
    # True SDF for a sphere of radius 0.5
    # SDF = ||coords|| - radius
    radius = 0.5
    gt_sdf = torch.norm(coords, dim=1, keepdim=True) - radius
    return coords, gt_sdf

def train_sdf():
    device = torch.device('cuda')
    print(f"--- Training FusionKAN SDF on {torch.cuda.get_device_name(0)} ---")
    
    # Model Setup
    # Grid size 32 is roughly equivalent to a multi-resolution grid
    model = FusionKANSDF(hidden_dim=32, grid_size=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda') # AMP Scaler
    
    steps = 2000
    batch_size = 16384
    
    start_time = time.time()
    
    loss_history = []
    
    for step in range(steps):
        # 1. Get Data
        coords, gt_sdf = get_sphere_data(batch_size, device)
        
        optimizer.zero_grad()
        
        # 2. Forward (Mixed Precision)
        with autocast('cuda'):
            pred_sdf = model(coords)
            loss = torch.nn.functional.mse_loss(pred_sdf, gt_sdf)
        
        # 3. Backward & Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")
            loss_history.append(loss.item())
            
            # Simple Grid Adaptation Logic (Optional Research Direction)
            # If grids stretch too far, you might clamp them, but FusionKAN handles 
            # learnable bounds automatically in the backward pass.

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f}s")
    
    # --- VISUALIZATION ---
    # Slice at Z=0
    res = 256
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)
    
    grid_coords = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Inference can be FP32 or FP16
        pred = model(grid_tensor)
        
    pred_grid = pred.cpu().numpy().reshape(res, res)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Learned SDF (Z=0 Slice)")
    plt.imshow(pred_grid, extent=[-1,1,-1,1], origin='lower', cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Distance')
    plt.contour(xv, yv, pred_grid, levels=[0], colors='black') # Zero level set (surface)
    
    plt.subplot(1, 2, 2)
    plt.title("Loss Curve")
    plt.plot(loss_history)
    plt.yscale('log')
    
    plt.savefig('sdf_result.png')
    print("Result saved to sdf_result.png")

if __name__ == "__main__":
    train_sdf()