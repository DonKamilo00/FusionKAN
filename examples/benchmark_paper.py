import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import ellipj
import gc
import os
import sys

# --- Imports ---
# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fusion_kan import FusionKANLayer
except ImportError:
    raise ImportError("FusionKAN not installed. Run 'pip install .'")

try:
    from kan.MultKAN import MultKAN
except ImportError:
    print("⚠️ Original KAN not found in 'kan/' folder. Will skip original benchmark.")
    MultKAN = None

# --- Task: Jacobi Elliptic Function sn(x, m) ---
# The paper uses this as a hard test case for function approximation.
def generate_data(n_samples=10000, device='cuda'):
    # Domain: x in [0, 10], m in [0, 1]
    x = np.random.rand(n_samples) * 10.0
    m = np.random.rand(n_samples) # Parameter m
    
    # Ground Truth: sn(x, m)
    sn, cn, dn, ph = ellipj(x, m)
    y = sn
    
    # Normalize Inputs to [-1, 1] for KAN
    # x: [0, 10] -> [-1, 1] => (x - 5) / 5
    # m: [0, 1]  -> [-1, 1] => (m - 0.5) / 0.5
    input_data = np.stack([(x - 5.0)/5.0, (m - 0.5)/0.5], axis=1)
    
    inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
    labels = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
    return inputs, labels

# --- Benchmarking Logic ---
def train_model(name, model, x, y, steps=1000):
    print(f"\n🧠 Training {name}...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    
    torch.cuda.synchronize()
    start_mem = torch.cuda.max_memory_allocated()
    start_time = time.time()
    
    history = []
    
    for i in range(steps):
        opt.zero_grad()
        if "Original" in name:
            pred = model(x)
        else:
            pred = model(x)
            
        loss = torch.nn.MSELoss()(pred, y)
        loss.backward()
        opt.step()
        
        if i % 100 == 0:
            history.append(loss.item())
            # print(f"  Step {i}: {loss.item():.6f}")
            
        if i % 50 == 0:
            scheduler.step()
            
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_mem = (torch.cuda.max_memory_allocated() - start_mem) / 1024**2
    
    print(f"  > Final MSE: {history[-1]:.6f}")
    print(f"  > Time: {total_time:.2f}s")
    print(f"  > VRAM Delta: {peak_mem:.2f} MB")
    
    return history, total_time, peak_mem

def run_paper_benchmark():
    print("--- 📜 Reproducing KAN Paper Experiment: Elliptic Integral ---")
    DEVICE = 'cuda'
    
    # 1. Data
    # 50k samples (Paper uses dense grid, we use random sampling)
    train_x, train_y = generate_data(50000, DEVICE)
    test_x, test_y = generate_data(10000, DEVICE)
    
    # 2. Config (Paper uses [2, 2, 1] for this task)
    WIDTH = [2, 2, 1] 
    GRID = 20 # Paper often uses 5-20
    K = 3
    STEPS = 2000
    
    results = {}
    
    # 3. Models
    if MultKAN:
        torch.cuda.empty_cache()
        orig = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False).to(DEVICE)
        h_orig, t_orig, m_orig = train_model("Original KAN", orig, train_x, train_y, steps=STEPS)
        results['Original'] = {'loss': h_orig, 'time': t_orig, 'mem': m_orig}
        del orig

    torch.cuda.empty_cache()
    fused = torch.nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, spline_order=K),
        FusionKANLayer(2, 1, grid_size=GRID, spline_order=K, is_output=True)
    ).to(DEVICE)
    
    h_fused, t_fused, m_fused = train_model("FusionKAN", fused, train_x, train_y, steps=STEPS)
    results['Fused'] = {'loss': h_fused, 'time': t_fused, 'mem': m_fused}
    
    # 4. Visualization
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['loss'], label=f"{name} ({res['time']:.1f}s)")
    plt.yscale('log')
    plt.title("Convergence Speed (Elliptic Function)")
    plt.xlabel("Step (x100)")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Function Fit Visualization
    plt.subplot(1, 2, 2)
    # Create a slice where m=0.5 (Input index 1 = 0.0)
    x_slice = torch.linspace(-1, 1, 200, device=DEVICE)
    m_slice = torch.zeros_like(x_slice)
    inputs = torch.stack([x_slice, m_slice], dim=1)
    
    # Ground Truth
    # Un-normalize for scipy: x_real = x*5 + 5
    x_real = x_slice.cpu().numpy() * 5.0 + 5.0
    y_gt, _, _, _ = ellipj(x_real, 0.5)
    
    plt.plot(x_slice.cpu(), y_gt, 'k-', label='Ground Truth (m=0.5)', linewidth=2, alpha=0.6)
    
    with torch.no_grad():
        y_fused = fused(inputs)
        plt.plot(x_slice.cpu(), y_fused.cpu(), 'r--', label='FusionKAN Pred')
        
    plt.title("Function Approximation (Slice)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("paper_reproduction.png")
    print("\nSaved 'paper_reproduction.png'")
    plt.show()

if __name__ == "__main__":
    run_paper_benchmark()