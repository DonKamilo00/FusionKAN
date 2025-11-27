import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os

# Setup Path
sys.path.append(os.getcwd())
try:
    from fusion_kan import FusionKANLayer
except:
    # Fallback for local dev
    from fusion_kan.layer import FusionKANLayer

# Try importing Original KAN
try:
    from kan.MultKAN import MultKAN
except:
    MultKAN = None

def generate_data(n_samples=50000, device='cuda'):
    # Task: f(x,m) = sn(x, m)
    # x in [0, 10], m in [0, 1]
    x = np.random.rand(n_samples) * 10.0
    m = np.random.rand(n_samples)
    sn, _, _, _ = ellipj(x, m)
    
    # Normalize inputs to [-1, 1]
    x_n = (x - 5.0) / 5.0
    m_n = (m - 0.5) / 0.5
    
    inputs = torch.tensor(np.stack([x_n, m_n], 1), dtype=torch.float32).to(device)
    labels = torch.tensor(sn, dtype=torch.float32).to(device).unsqueeze(1)
    return inputs, labels

def train_loop(name, model, x, y, steps=2000):
    print(f"\n🧠 Training {name}...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99) # Decay to refine spline
    
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
    t = time.time() - start
    print(f"  > Final MSE: {history[-1]:.5f}")
    print(f"  > Time: {t:.2f}s")
    return history, t

def run_benchmark():
    print("--- 📜 Reproducing Paper Result (Elliptic Integral) ---")
    DEVICE = 'cuda'
    x, y = generate_data(50000, DEVICE)
    
    GRID = 50
    # Important: Slightly padded grid range to avoid boundary issues with normalized data
    RANGE = [-1.1, 1.1] 
    
    # 1. Train Original
    orig_hist = []
    if MultKAN:
        # Original KAN uses adaptive grid, so it handles [-1,1] fine.
        model_orig = MultKAN(width=[2,2,1], grid=20, k=3, symbolic_enabled=False, device=DEVICE)
        orig_hist, orig_t = train_loop("Original", model_orig, x, y)

    # 2. Train Fused
    model_fused = nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, spline_order=3, grid_range=RANGE),
        FusionKANLayer(2, 1, grid_size=GRID, spline_order=3, grid_range=RANGE, is_output=True)
    ).to(DEVICE)
    
    fused_hist, fused_t = train_loop("Fused", model_fused, x, y)
    
    # 3. Visuals
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    if orig_hist: plt.plot(orig_hist, label=f"Original ({orig_t:.1f}s)")
    plt.plot(fused_hist, label=f"Fused ({fused_t:.1f}s)", linestyle='--')
    plt.yscale('log')
    plt.title("Convergence (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Slice Fit (m=0.5)
    plt.subplot(1, 2, 2)
    x_raw = np.linspace(0, 10, 200)
    # m = 0.5 corresponds to normalized 0.0
    inp = torch.tensor(np.stack([(x_raw-5)/5, np.zeros_like(x_raw)], 1), dtype=torch.float32).to(DEVICE)
    
    gt, _, _, _ = ellipj(x_raw, 0.5)
    with torch.no_grad():
        pred = model_fused(inp).cpu().numpy()
        
    plt.plot(x_raw, gt, 'k', label='Ground Truth', alpha=0.5, linewidth=2)
    plt.plot(x_raw, pred, 'r--', label='Fused Pred')
    plt.title("Fit (m=0.5)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("paper_benchmark_fixed.png")
    print("Saved 'paper_benchmark_fixed.png'")
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_benchmark()