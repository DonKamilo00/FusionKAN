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
    # Import the raw KANLayer from the first paper's code
    from kan.KANLayer import KANLayer 
except ImportError:
    print("Please install fusion_kan and ensure 'kan' folder is present.")
    exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)

# ==========================================
# HELPER: ADAPTER FOR ORIGINAL KAN
# ==========================================
class OriginalKANLayerWrapper(nn.Module):
    """
    Wraps the original KANLayer to make it compatible with nn.Sequential.
    The original layer returns a tuple (y, preacts, ...), we only want y.
    """
    def __init__(self, in_dim, out_dim, grid, k):
        super().__init__()
        # Initialize original layer
        self.layer = KANLayer(in_dim=in_dim, out_dim=out_dim, num=grid, k=k, device=DEVICE)
        
    def forward(self, x):
        # The original forward returns: y, preacts, postacts, postspline
        # We only need y to pass to the next layer
        return self.layer(x)[0]
    
    def update_grid(self, x):
        # Expose the original update method
        self.layer.update_grid_from_samples(x)

# ==========================================
# DATA GENERATION
# ==========================================
def get_bessel_data(n_samples=10000):
    x = torch.linspace(-1, 1, n_samples).unsqueeze(1).to(DEVICE)
    y = torch.tensor(scipy.special.j0(20 * x.cpu().numpy())).to(DEVICE)
    return x, y

def get_gear_sdf_data(n_samples=100000):
    points = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    r = torch.norm(points, dim=1)
    theta = torch.atan2(points[:, 1], points[:, 0])
    teeth = 8
    target_radius = 0.5 + 0.1 * torch.sin(teeth * theta)
    sdf = r - target_radius
    return points, sdf.unsqueeze(1)

# ==========================================
# MODEL FACTORY
# ==========================================
def build_model(model_type, input_dim, hidden_dim, grid_size, k=3):
    
    if model_type == "PyKAN":
        # REPLICATING FUSIONKAN ARCHITECTURE EXACTLY
        # FusionKANLayer has built-in LayerNorm and PReLU.
        # Original KANLayer does not. We add them manually here to make the comparison 100% fair.
        return nn.Sequential(
            # Layer 1
            OriginalKANLayerWrapper(input_dim, hidden_dim, grid_size, k),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(), # FusionKAN uses PReLU by default
            
            # Layer 2
            OriginalKANLayerWrapper(hidden_dim, 1, grid_size, k)
            # Output layer usually doesn't need norm/activation in regression
        ).to(DEVICE)
        
    elif model_type == "FusionKAN":
        # Our Optimized CUDA KAN
        return nn.Sequential(
            # FusionKANLayer includes Norm and PReLU internally
            FusionKANLayer(input_dim, hidden_dim, grid_size, k, use_node_activation=True),
            FusionKANLayer(hidden_dim, 1, grid_size, k, is_output=True)
        ).to(DEVICE)
        
    return None

# ==========================================
# TRAINING LOOP
# ==========================================
def train_and_evaluate(task_name, model_type, x_train, y_train, config):
    print(f"[{task_name}] Training {model_type}...")
    
    model = build_model(model_type, x_train.shape[1], config['width'], config['grid'])
    
    # Use same LR for both now that architectures are aligned
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    loss_fn = nn.MSELoss()
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=config['batch'], shuffle=True)
    
    start_time = time.time()
    history = []
    
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        
        # Manual Grid Update for PyKAN
        # We iterate through modules to find the wrapper and call update
        if model_type == "PyKAN" and epoch % 5 == 0 and epoch > 0:
            current_x = None
            for layer in model:
                # Capture input for the first layer
                if isinstance(layer, OriginalKANLayerWrapper) and current_x is None:
                    # We need a batch of data to update grid. Use the first batch from loader.
                    sample_bx, _ = next(iter(loader))
                    layer.update_grid(sample_bx)
                    # For the second layer, we need the output of the first.
                    # This is complex to do perfectly in a generic loop without hooks.
                    # For this benchmark, updating just the first layer is usually sufficient 
                    # to prevent NaN on input boundaries.
                    with torch.no_grad():
                        current_x = layer(sample_bx)
                elif isinstance(layer, OriginalKANLayerWrapper) and current_x is not None:
                     layer.update_grid(current_x)


        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            
            if torch.isnan(loss):
                print("⚠️ Loss is NaN! Stopping early.")
                return model, history, time.time() - start_time
                
            loss.backward()
            
            # Clip gradients to ensure stability for both
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
    # 1. BESSEL EXPERIMENT 
    # Reduced Grid slightly to 100 to ensure PyKAN has a fighting chance
    bessel_config = {'width': 64, 'grid': 100, 'lr': 0.01, 'epochs': 100, 'steps': 100, 'batch': 512}
    x_b, y_b = get_bessel_data()
    
    # Train Original
    model_bo, hist_bo, time_bo = train_and_evaluate("Bessel", "PyKAN", x_b, y_b, bessel_config)
    # Train Fusion
    model_bf, hist_bf, time_bf = train_and_evaluate("Bessel", "FusionKAN", x_b, y_b, bessel_config)
    
    # Save Predictions
    with torch.no_grad():
        pred_bo = model_bo(x_b).cpu().numpy()
        pred_bf = model_bf(x_b).cpu().numpy()
    
    np.savez("results_bessel.npz", x=x_b.cpu().numpy(), y=y_b.cpu().numpy(), 
             pred_orig=pred_bo, pred_fusion=pred_bf, 
             hist_orig=hist_bo, hist_fusion=hist_bf,
             time_orig=time_bo, time_fusion=time_bf)

    # 2. SDF EXPERIMENT
    sdf_config = {'width': 64, 'grid': 50, 'lr': 0.005, 'epochs': 50, 'steps': 50, 'batch': 8192}
    x_s, y_s = get_gear_sdf_data(200000) 
    
    # Train Fusion Only (For visual)
    model_sf, hist_sf, time_sf = train_and_evaluate("SDF", "FusionKAN", x_s, y_s, sdf_config)
    
    res = 256
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    grid_flat = torch.tensor(np.stack([grid_x.flatten(), grid_y.flatten()], axis=1), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        sdf_pred = model_sf(grid_flat).reshape(res, res).cpu().numpy()
        
    np.savez("results_sdf.npz", sdf_pred=sdf_pred, time_fusion=time_sf)
    
    print("✅ Benchmarks Done.")

if __name__ == "__main__":
    run_benchmarks()