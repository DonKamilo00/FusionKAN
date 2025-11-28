import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import gc
import os
import sys

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Import FusionKAN
try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported successfully.")
except ImportError:
    raise ImportError("Please install fusion_kan first via 'pip install .'")

# 2. Import Original KAN (MultKAN class handles both versions)
try:
    from kan.MultKAN import MultKAN
    print("✅ Original KAN code imported successfully.")
except ImportError:
    print("⚠️ Could not import original 'kan' package.")
    MultKAN = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = "fusion_kan_paper_results.csv"

# --- DATA GENERATION ---
def get_data(n_samples=10000):
    # Task: f(x,y) = exp(sin(pi*x) + y^2)
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

# --- TRAINING HARNESS ---
def run_trial(model_type, config, x_train, y_train):
    width = config['width']
    grid = config['grid']
    k = 3
    batch_size = config['batch']
    steps = 100 # Short run for speed/mem
    if config['type'] == 'convergence': steps = 1000
    
    # 1. Initialize Model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    try:
        if model_type == "PyKAN":
            # ORIGINAL KAN (Paper 1)
            # Pure splines, no multiplication nodes.
            # mult_arity=0 forces the original behavior.
            model = MultKAN(width=[2, width, 1], grid=grid, k=k, 
                           mult_arity=0, 
                           symbolic_enabled=False, device=DEVICE)

        elif model_type == "MultKAN":
            # KAN 2.0 (Paper 2)
            # Includes multiplication nodes.
            # Width format is [[n_sum, n_mult], ...]
            # We add 5 multiplication nodes to the hidden layer for comparison
            m_width = [[2,0], [width, 5], [1,0]] 
            model = MultKAN(width=m_width, grid=grid, k=k, 
                           mult_arity=2, 
                           symbolic_enabled=False, device=DEVICE)
            
        else: 
            # FUSION KAN (Ours)
            model = nn.Sequential(
                FusionKANLayer(2, width, grid_size=grid, spline_order=k),
                FusionKANLayer(width, 1, grid_size=grid, spline_order=k, is_output=True)
            ).to(DEVICE)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Warmup (Optional, mainly for JIT compilation)
        # _ = model(x_train[:10])
        torch.cuda.synchronize()
        
        # 2. Training Loop
        start_time = time.time()
        losses = []
        
        for step in range(steps):
            indices = torch.randperm(x_train.size(0))[:batch_size]
            bx, by = x_train[indices], y_train[indices]
            
            # Both PyKAN and MultKAN require manual grid updates
            if model_type in ["PyKAN", "MultKAN"] and step % 50 == 0 and step > 0:
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
            "loss_curve": losses
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {"status": "OOM"}
        else:
            return {"status": "error", "msg": str(e)}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# --- EXPERIMENTS ---

def run_experiments():
    results = []
    x, y = get_data(20000)
    
    # List of models to compare
    # PyKAN = Original KAN (Paper 1)
    # MultKAN = KAN 2.0 (Paper 2)
    models_to_test = ["FusionKAN", "PyKAN", "MultKAN"]
    
    print("--- 1. GRID SCALING EXPERIMENT (Memory Complexity) ---")
    grids = [10, 50, 100, 200, 500]
    fixed_width = 256
    fixed_batch = 4096
    
    for g in grids:
        conf = {'width': fixed_width, 'grid': g, 'batch': fixed_batch, 'type': 'grid_exp'}
        print(f"--- Running Grid={g} ---")
        
        for m_name in models_to_test:
            # Skip MultKAN for grid scaling if you want to save time (it behaves like PyKAN for memory)
            if m_name == "MultKAN" and g > 50: continue 

            print(f"Testing {m_name}...")
            res = run_trial(m_name, conf, x, y)
            
            if res['status'] == 'success':
                print(f"  > Time: {res['time']:.2f}s | Mem: {res['peak_mem_mb']:.0f}MB")
            else:
                print(f"  > {res['status']}")
            
            results.append({**conf, "model": m_name, **res})

    print("\n--- 2. WIDTH SCALING EXPERIMENT (Computational Throughput) ---")
    widths = [64, 128, 256, 512, 1024, 2048]
    fixed_grid = 20
    
    for w in widths:
        conf = {'width': w, 'grid': fixed_grid, 'batch': fixed_batch, 'type': 'width_exp'}
        print(f"--- Running Width={w} ---")
        
        for m_name in models_to_test:
            print(f"Testing {m_name}...")
            res = run_trial(m_name, conf, x, y)
            
            if res['status'] == 'success':
                print(f"  > Time: {res['time']:.2f}s | Speed: {res['ms_per_step']:.2f}ms/step")
            else:
                print(f"  > {res['status']}")
            
            results.append({**conf, "model": m_name, **res})

    print("\n--- 3. CONVERGENCE CHECK ---")
    # Smaller batch for convergence to allow more steps
    conf = {'width': 64, 'grid': 20, 'batch': 1024, 'type': 'convergence'}
    
    for m_name in models_to_test:
        print(f"Running Convergence {m_name}...")
        res = run_trial(m_name, conf, x, y)
        if res['status'] == 'success':
            np.save(f"loss_{m_name}.npy", res['loss_curve'])

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ All Benchmarks Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiments()