import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

# --- Imports ---
# 1. Import FusionKAN (Your Library)
try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported successfully.")
except ImportError:
    raise ImportError("Please install fusion_kan first via 'pip install .'")

# 2. Import Original KAN (From provided local files)
# Assuming the provided files are in a folder named 'kan' in the python path
try:
    from kan.MultKAN import MultKAN
    print("✅ Original KAN code imported successfully.")
except ImportError:
    print("⚠️ Could not import original 'kan' package. Please ensure the provided files are in a folder named 'kan'.")
    # Mocking for demonstration if files aren't set up, but in your case, they will be.
    MultKAN = None

# --- Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    print("⚠️ WARNING: Running on CPU. FusionKAN requires CUDA for speedups.")

def generate_data(n_samples=10000):
    # Task: f(x, y) = exp(sin(pi*x) + y^2)
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

# --- Training Helper ---
def train_model(model_name, model, x, y, steps=500, batch_size=512):
    print(f"\n🔄 Training {model_name}...")
    
    # Optimizer
    # We use Adam for fair throughput comparison (LBFGS behavior differs too much between implementations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    # Stats
    losses = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for step in range(steps):
        # Batching
        indices = torch.randperm(x.shape[0])[:batch_size]
        batch_x = x[indices]
        batch_y = y[indices]
        
        optimizer.zero_grad()
        
        # Forward
        # Original KAN uses a specific call signature, FusionKAN uses standard nn.Module
        if "Original" in model_name:
            pred = model(batch_x) 
        else:
            pred = model(batch_x)
            
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.5f}")

    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    
    print(f"✅ {model_name} Finished.")
    print(f"Time: {total_time:.4f}s | Peak VRAM: {peak_mem:.2f} MB")
    
    return total_time, peak_mem, losses

# --- Main Execution ---
def run_benchmark():
    # Hyperparams
    WIDTH = [2, 64, 1] # Input 2, Hidden 64, Output 1
    GRID = 10
    K = 3
    STEPS = 500
    BATCH = 2048
    
    # Data
    x, y = generate_data(n_samples=20000)
    
    results = {}
    
    # 1. Train Original KAN
    if MultKAN is not None:
        # symbolic_enabled=False ensures we are benchmarking the numerical engine only
        orig_model = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False, device=DEVICE)
        t_orig, m_orig, l_orig = train_model("Original KAN", orig_model, x, y, steps=STEPS, batch_size=BATCH)
        results['Original KAN'] = {'time': t_orig, 'mem': m_orig, 'loss': l_orig}
        
        # Cleanup
        del orig_model
        torch.cuda.empty_cache()
        gc.collect()

    # 2. Train FusionKAN
    # FusionKAN is layer-based, so we build a Sequential model to match the topology
    fused_model = nn.Sequential(
        FusionKANLayer(in_features=2, out_features=64, grid_size=GRID, spline_order=K),
        FusionKANLayer(in_features=64, out_features=1, grid_size=GRID, spline_order=K) # is_output=False to match MultKAN internal logic
    ).to(DEVICE)
    
    # Warmup to compile JIT kernels if not pre-compiled
    _ = fused_model(x[:10])
    
    t_fused, m_fused, l_fused = train_model("FusionKAN", fused_model, x, y, steps=STEPS, batch_size=BATCH)
    results['FusionKAN'] = {'time': t_fused, 'mem': m_fused, 'loss': l_fused}

    # --- Plotting ---
    plot_results(results)

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    mems = [results[n]['mem'] for n in names]
    colors = ['#FF5733', '#33FF57']
    
    # 1. Training Time
    ax1.bar(names, times, color=colors)
    ax1.set_title('Training Time (Lower is Better)')
    ax1.set_ylabel('Seconds')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}s", ha='center', va='bottom', fontweight='bold')
        
    # Speedup Annotation
    if len(names) > 1:
        speedup = times[0] / times[1]
        ax1.text(0.5, max(times)*0.5, f"{speedup:.1f}x Speedup", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # 2. Memory
    ax2.bar(names, mems, color=colors)
    ax2.set_title('Peak VRAM Usage (Lower is Better)')
    ax2.set_ylabel('MB')
    for i, v in enumerate(mems):
        ax2.text(i, v, f"{v:.0f} MB", ha='center', va='bottom', fontweight='bold')

    # 3. Loss Curves
    for name in names:
        ax3.plot(results[name]['loss'], label=name)
    ax3.set_title('Convergence (Loss)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('MSE Loss')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\n📊 Benchmark plot saved to 'benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()