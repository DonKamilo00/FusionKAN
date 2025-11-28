import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import gc
import os
import sys

# --- 🔧 FIX IMPORTS ---
# Add the project root directory to Python path so we can find 'kan' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Import FusionKAN (Your Library)
try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported successfully.")
except ImportError:
    raise ImportError("Please install fusion_kan first via 'pip install .'")

# 2. Import Original KAN (From local folder)
# Assuming the provided files are in a folder named 'kan' in the python path
try:
    from kan.MultKAN import MultKAN
    print("✅ Original KAN code imported successfully.")
except ImportError:
    print("⚠️ Could not import original 'kan' package. Please ensure the provided files are in a folder named 'kan'.")
    # Mocking for demonstration if files aren't set up, but in your case, they will be.
    MultKAN = None

DEVICE = 'cuda'
RESULTS_FILE = "fusion_kan_paper_results.csv"

# --- DATA GENERATION ---
def get_data(n_samples=10000):
    # Task: f(x,y) = exp(sin(pi*x) + y^2)
    # A standard smooth function from the KAN paper
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

# --- TRAINING HARNESS ---
def run_trial(model_type, config, x_train, y_train):
    width = config['width']
    grid = config['grid']
    k = 3
    batch_size = config['batch']
    steps = 100 # Short run for speed/mem, Long run for convergence
    if config['type'] == 'convergence': steps = 1000
    
    # 1. Initialize Model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    try:
        if model_type == "Original":
            # symbolic_enabled=False for fair numeric comparison
            model = MultKAN(width=[2, width, 1], grid=grid, k=k, 
                           symbolic_enabled=False, device=DEVICE)
        else:
            model = nn.Sequential(
                FusionKANLayer(2, width, grid, k),
                FusionKANLayer(width, 1, grid, k, is_output=True)
            ).to(DEVICE)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Warmup
        _ = model(x_train[:10])
        torch.cuda.synchronize()
        
        # 2. Training Loop
        start_time = time.time()
        losses = []
        
        for step in range(steps):
            indices = torch.randperm(x_train.size(0))[:batch_size]
            bx, by = x_train[indices], y_train[indices]
            
            # Original KAN manual grid update (essential for convergence)
            if model_type == "Original" and step % 50 == 0 and step > 0:
                model.update_grid(bx)
                
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
            if config['type'] == 'convergence':
                losses.append(loss.item())
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 3. Metrics
        total_time = end_time - start_time
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        avg_iter_ms = (total_time / steps) * 1000
        final_loss = loss.item()
        
        del model
        return {
            "status": "success",
            "time": total_time,
            "ms_per_step": avg_iter_ms,
            "peak_mem_mb": peak_mem_mb,
            "final_loss": final_loss,
            "loss_curve": losses # Only for convergence test
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {"status": "OOM"}
        else:
            return {"status": "error", "msg": str(e)}

# --- EXPERIMENTS ---

def run_experiments():
    results = []
    x, y = get_data(20000)
    
    print("--- 1. GRID SCALING EXPERIMENT (Memory Complexity) ---")
    # Hypothesis: Original KAN memory explodes with Grid size. FusionKAN is constant.
    grids = [10, 50, 100, 200, 500]
    fixed_width = 256
    fixed_batch = 4096
    
    for g in grids:
        conf = {'width': fixed_width, 'grid': g, 'batch': fixed_batch, 'type': 'grid_exp'}
        
        print(f"Running Grid={g}...")
        
        # Fusion
        res_f = run_trial("Fusion", conf, x, y)
        results.append({**conf, "model": "FusionKAN", **res_f})
        
        # Original
        res_o = run_trial("Original", conf, x, y)
        results.append({**conf, "model": "Original", **res_o})

    print("\n--- 2. WIDTH SCALING EXPERIMENT (Computational Throughput) ---")
    # Hypothesis: FusionKAN has much lower latency per step at large widths.
    widths = [64, 128, 256, 512, 1024, 2048]
    fixed_grid = 20
    
    for w in widths:
        conf = {'width': w, 'grid': fixed_grid, 'batch': fixed_batch, 'type': 'width_exp'}
        
        print(f"Running Width={w}...")
        
        res_f = run_trial("Fusion", conf, x, y)
        results.append({**conf, "model": "FusionKAN", **res_f})
        
        res_o = run_trial("Original", conf, x, y)
        results.append({**conf, "model": "Original", **res_o})

    print("\n--- 3. CONVERGENCE CHECK ---")
    # Hypothesis: FusionKAN (Learnable Grid) converges as smooth/fast as Original (Adaptive Grid)
    conf = {'width': 64, 'grid': 20, 'batch': 1024, 'type': 'convergence'}
    
    print("Running Convergence Fusion...")
    res_f = run_trial("Fusion", conf, x, y)
    
    print("Running Convergence Original...")
    res_o = run_trial("Original", conf, x, y)
    
    # Save loss curves separately
    np.save("loss_fusion.npy", res_f['loss_curve'])
    np.save("loss_original.npy", res_o['loss_curve'])

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ All Benchmarks Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiments()