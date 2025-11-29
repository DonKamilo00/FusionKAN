import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

# --- IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from fusion_kan import FusionKANLayer
    from kan.MultKAN import MultKAN
except ImportError:
    print("Please install fusion_kan and ensure 'kan' folder is present.")
    exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)

# ==========================================
# TASK 1: BESSEL FUNCTION (PAPER 1)
# ==========================================
def get_bessel_data(n_samples=10000):
    # f(x) = J0(20x), highly oscillatory
    x = torch.linspace(-1, 1, n_samples).unsqueeze(1).to(DEVICE)
    y = torch.tensor(scipy.special.j0(20 * x.cpu().numpy())).to(DEVICE)
    return x, y

# ==========================================
# TASK 2: 2D GEAR SDF (GRAPHICS)
# ==========================================
def get_gear_sdf_data(n_samples=100000):
    # Create a complex 2D shape (Gear)
    points = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    
    # Convert to polar
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    
    # Gear formula: R + variation * sin(teeth * theta)
    teeth = 8
    target_radius = 0.5 + 0.1 * torch.sin(teeth * theta)
    
    # SDF = dist to center - target_radius
    sdf = r - target_radius
    
    return points, sdf.unsqueeze(1)

# ==========================================
# MODEL FACTORY
# ==========================================
def build_model(model_type, input_dim, hidden_dim, grid_size, k=3):
    if model_type == "PyKAN":
        # Standard KAN from Paper 1 (No Mul)
        # mult_arity=0 forces pure B-spline layers
        return MultKAN(width=[input_dim, hidden_dim, 1], grid=grid_size, k=k, 
                      mult_arity=0, symbolic_enabled=False, device=DEVICE)
    
    elif model_type == "FusionKAN":
        # Our CUDA KAN
        return nn.Sequential(
            FusionKANLayer(input_dim, hidden_dim, grid_size, k),
            FusionKANLayer(hidden_dim, 1, grid_size, k, is_output=True)
        ).to(DEVICE)
    return None

# ==========================================
# TRAINING LOOP
# ==========================================
def train_and_evaluate(task_name, model_type, x_train, y_train, config):
    print(f"[{task_name}] Training {model_type}...")
    
    model = build_model(model_type, x_train.shape[1], config['width'], config['grid'])
    
    # --- HYPERPARAMETER ADJUSTMENT ---
    # PyKAN is numerically unstable at high grid sizes (G=200).
    # We reduce LR by 10x to prevent NaN divergence.
    lr = config['lr']
    if model_type == "PyKAN":
        lr = lr * 0.1
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    loss_fn = nn.MSELoss()
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=config['batch'], shuffle=True)
    
    start_time = time.time()
    history = []
    
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for bx, by in loader:
            # Manual Grid Update for PyKAN (Essential for adaptation)
            if model_type == "PyKAN" and epoch % 5 == 0 and epoch > 0:
                model.update_grid(bx)
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            
            if torch.isnan(loss):
                print("⚠️ Loss is NaN! Stopping early.")
                break
                
            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            # Essential for high-grid B-splines to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Ep {epoch}: Loss {avg_loss:.6f}")
            
    total_time = time.time() - start_time
    
    return model, history, total_time

# ==========================================
# MAIN EXPERIMENTS
# ==========================================

def run_benchmarks():
    # 1. BESSEL EXPERIMENT (High Frequency Fitting)
    # Grid=200 is very high resolution. FusionKAN handles this easily via CUDA.
    # PyKAN needs clipping and lower LR to survive.
    bessel_config = {'width': 64, 'grid': 200, 'lr': 0.01, 'epochs': 100, 'steps': 100, 'batch': 512}
    x_b, y_b = get_bessel_data()
    
    # Train Original
    model_bo, hist_bo, time_bo = train_and_evaluate("Bessel", "PyKAN", x_b, y_b, bessel_config)
    # Train Fusion
    model_bf, hist_bf, time_bf = train_and_evaluate("Bessel", "FusionKAN", x_b, y_b, bessel_config)
    
    # Save Predictions for Plotting
    with torch.no_grad():
        pred_bo = model_bo(x_b).cpu().numpy()
        pred_bf = model_bf(x_b).cpu().numpy()
    
    np.savez("results_bessel.npz", x=x_b.cpu().numpy(), y=y_b.cpu().numpy(), 
             pred_orig=pred_bo, pred_fusion=pred_bf, 
             hist_orig=hist_bo, hist_fusion=hist_bf,
             time_orig=time_bo, time_fusion=time_bf)

    # 2. SDF EXPERIMENT (2D Shape Representation)
    # Large batch size (8192) to stress throughput.
    sdf_config = {'width': 64, 'grid': 50, 'lr': 0.005, 'epochs': 50, 'steps': 50, 'batch': 8192}
    x_s, y_s = get_gear_sdf_data(200000) # 200k points
    
    # Train Fusion Only (PyKAN is too slow for this loop in reasonable time)
    model_sf, hist_sf, time_sf = train_and_evaluate("SDF", "FusionKAN", x_s, y_s, sdf_config)
    
    # Generate Inference Grid for Visualization
    res = 256
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    grid_flat = torch.tensor(np.stack([grid_x.flatten(), grid_y.flatten()], axis=1), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        sdf_pred = model_sf(grid_flat).reshape(res, res).cpu().numpy()
        
    np.savez("results_sdf.npz", sdf_pred=sdf_pred, time_fusion=time_sf)
    
    print("✅ Benchmarks Done.")

if __name__ == "__main__":
    run_benchmarks()