import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os
import gc

# Fix imports
sys.path.append(os.getcwd())
from fusion_kan import FusionKANLayer
# Try importing Original KAN if available
try:
    from kan.MultKAN import MultKAN
except:
    MultKAN = None

def generate_data(n_samples=50000, device='cuda'):
    x = np.random.rand(n_samples) * 10.0
    m = np.random.rand(n_samples)
    sn, _, _, _ = ellipj(x, m)
    
    # Normalize [0, 10] -> [-1, 1]
    x_n = (x - 5.0) / 5.0
    # Normalize [0, 1] -> [-1, 1]
    m_n = (m - 0.5) / 0.5
    
    inputs = torch.tensor(np.stack([x_n, m_n], 1), dtype=torch.float32).to(device)
    labels = torch.tensor(sn, dtype=torch.float32).to(device).unsqueeze(1)
    return inputs, labels

def train_loop(model, x, y, steps=2000):
    # Use same optimizer settings as Original KAN paper (usually LBFGS, but Adam is standard for comparison)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.995)
    
    history = []
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        
        if i % 50 == 0:
            history.append(loss.item())
            sch.step()
            
    torch.cuda.synchronize()
    total = time.time() - start
    return history, total

def run_paper_benchmark():
    print("--- 🧪 Final Paper Benchmark (Elliptic Integral) ---")
    DEVICE = 'cuda'
    x, y = generate_data(50000, DEVICE)
    
    GRID = 50
    RANGE = [-1.2, 1.2] # Padded
    
    results = {}
    
    # 1. Original
    if MultKAN:
        print("Training Original...")
        torch.cuda.empty_cache()
        # Original KAN initialization is handled internally by MultKAN
        orig = MultKAN(width=[2,2,1], grid=20, k=3, symbolic_enabled=False, device=DEVICE)
        h_orig, t_orig = train_loop(orig, x, y)
        results['Original'] = (h_orig, t_orig)
    
    # 2. Fused
    print("Training Fused...")
    torch.cuda.empty_cache()
    fused = nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, grid_range=RANGE, spline_order=3),
        FusionKANLayer(2, 1, grid_size=GRID, grid_range=RANGE, spline_order=3, is_output=True)
    ).to(DEVICE)
    
    h_fused, t_fused = train_loop(fused, x, y)
    results['Fused'] = (h_fused, t_fused)
    
    # 3. Plot
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    for name, (h, t) in results.items():
        plt.plot(h, label=f"{name} ({t:.1f}s)")
    plt.yscale('log')
    plt.title(f"Convergence (MSE) - Final MSE: {h_fused[-1]:.5f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Slice
    plt.subplot(1, 2, 2)
    x_raw = np.linspace(0, 10, 200)
    m_val = 0.5
    inp = torch.tensor(np.stack([(x_raw-5)/5, np.full_like(x_raw, (m_val-0.5)/0.5)], 1), dtype=torch.float32).cuda()
    
    gt, _, _, _ = ellipj(x_raw, m_val)
    with torch.no_grad():
        pred = fused(inp).cpu().numpy()
        
    plt.plot(x_raw, gt, 'k', label='Ground Truth', alpha=0.5, linewidth=2)
    plt.plot(x_raw, pred, 'r--', label='Fused Pred')
    plt.title("Function Fit (m=0.5)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("final_paper_benchmark.png")
    print("Saved 'final_paper_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_paper_benchmark()