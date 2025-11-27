import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
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
try:
    from kan.MultKAN import MultKAN
    print("✅ Original KAN code imported successfully.")
except ImportError as e:
    print(f"⚠️ Error importing Original KAN: {e}")
    print("Ensure you have 'sympy' and 'scikit-learn' installed: `pip install sympy scikit-learn`")
    MultKAN = None

# --- Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {DEVICE}")

def generate_data(n_samples=10000):
    # Task: f(x, y) = exp(sin(pi*x) + y^2)
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

# --- Training Helper ---
def train_model(model_name, model, x, y, steps=500, batch_size=512):
    print(f"\n🔄 Training {model_name}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    losses = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for step in range(steps):
        indices = torch.randperm(x.shape[0])[:batch_size]
        batch_x = x[indices]
        batch_y = y[indices]
        
        optimizer.zero_grad()
        
        # Original KAN implementation might need specific handling
        if "Original" in model_name:
            try:
                # Original KAN forward often expects just x, but let's be safe
                pred = model(batch_x)
            except Exception as e:
                # Some KAN implementations return tuples or behave differently
                print(f"Forward pass error in Original KAN: {e}")
                return 0, 0, []
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
    # Hyperparams from Paper
    WIDTH = [2, 64, 1] 
    GRID = 10
    K = 3
    STEPS = 500
    BATCH = 2048
    
    x, y = generate_data(n_samples=20000)
    
    results = {}
    
    # 1. Train Original KAN
    if MultKAN is not None:
        try:
            # Initialize Original
            orig_model = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False, device=DEVICE)
            
            t_orig, m_orig, l_orig = train_model("Original KAN", orig_model, x, y, steps=STEPS, batch_size=BATCH)
            
            if t_orig > 0:
                results['Original KAN'] = {'time': t_orig, 'mem': m_orig, 'loss': l_orig}
            
            del orig_model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Original KAN failed to initialize/train: {e}")

    # 2. Train FusionKAN
    fused_model = nn.Sequential(
        FusionKANLayer(in_features=2, out_features=64, grid_size=GRID, spline_order=K),
        FusionKANLayer(in_features=64, out_features=1, grid_size=GRID, spline_order=K, is_output=True)
    ).to(DEVICE)
    
    # Warmup
    _ = fused_model(x[:10])
    
    t_fused, m_fused, l_fused = train_model("FusionKAN", fused_model, x, y, steps=STEPS, batch_size=BATCH)
    results['FusionKAN'] = {'time': t_fused, 'mem': m_fused, 'loss': l_fused}

    # --- Plotting ---
    if len(results) > 0:
        plot_results(results)
    else:
        print("No results to plot.")

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    mems = [results[n]['mem'] for n in names]
    colors = ['#FF5733', '#33FF57']
    
    # 1. Training Time
    ax1.bar(names, times, color=colors[:len(names)])
    ax1.set_title('Training Time (Lower is Better)')
    ax1.set_ylabel('Seconds')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}s", ha='center', va='bottom', fontweight='bold')
        
    if "Original KAN" in results and "FusionKAN" in results:
        speedup = results['Original KAN']['time'] / results['FusionKAN']['time']
        ax1.text(0.5, max(times)*0.5, f"{speedup:.1f}x Speedup", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # 2. Memory
    ax2.bar(names, mems, color=colors[:len(names)])
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