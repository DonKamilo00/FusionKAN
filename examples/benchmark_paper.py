import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import ellipj
import sys
import os

# Fix path
sys.path.append(os.getcwd())
from fusion_kan import FusionKANLayer
from kan.MultKAN import MultKAN

# --- Task: Jacobi Elliptic Function ---
def generate_data(n_samples=10000, device='cuda'):
    x = np.random.rand(n_samples) * 10.0
    m = np.random.rand(n_samples)
    sn, cn, dn, ph = ellipj(x, m)
    y = sn
    # Normalize inputs to strictly [-1, 1]
    input_data = np.stack([(x - 5.0)/5.0, (m - 0.5)/0.5], axis=1)
    inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
    labels = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
    return inputs, labels

def train_model(name, model, x, y, steps=1000):
    print(f"\n🧠 Training {name}...")
    # Lower LR for stability
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    torch.cuda.synchronize()
    start = time.time()
    
    history = []
    for i in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = torch.nn.MSELoss()(pred, y)
        loss.backward()
        opt.step()
        history.append(loss.item())
        
    torch.cuda.synchronize()
    total_time = time.time() - start
    print(f"  > Final MSE: {history[-1]:.6f}")
    print(f"  > Time: {total_time:.2f}s")
    return history, total_time

def run_paper_benchmark_v2():
    print("--- 📜 Reproducing KAN Paper Experiment (Fixed) ---")
    DEVICE = 'cuda'
    train_x, train_y = generate_data(50000, DEVICE)
    
    # Config
    # Paper uses Grid=5 initially and extends to 100.
    # We will start with Grid=20 to give FusionKAN enough capacity immediately.
    WIDTH = [2, 2, 1]
    GRID = 50 # Increased from 20 to ensure convergence without grid adaptivity
    K = 3
    STEPS = 2000
    
    results = {}
    
    # 1. Original
    print("Initializing Original KAN...")
    orig = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False, device=DEVICE)
    h_orig, t_orig = train_model("Original KAN", orig, train_x, train_y, steps=STEPS)
    results['Original'] = {'loss': h_orig, 'time': t_orig}

    # 2. FusionKAN
    print("Initializing FusionKAN...")
    # Important: Grid Range [-1, 1] must match data normalization exactly
    fused = torch.nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, spline_order=K, grid_range=[-1, 1]),
        FusionKANLayer(2, 1, grid_size=GRID, spline_order=K, grid_range=[-1, 1], is_output=True)
    ).to(DEVICE)
    
    h_fused, t_fused = train_model("FusionKAN", fused, train_x, train_y, steps=STEPS)
    results['Fused'] = {'loss': h_fused, 'time': t_fused}
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['loss'], label=f"{name}")
    plt.yscale('log')
    plt.title("MSE Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    x_slice = torch.linspace(-1, 1, 200, device=DEVICE)
    m_slice = torch.zeros_like(x_slice) # m = 0.5 (normalized is 0)
    inputs = torch.stack([x_slice, m_slice], dim=1)
    
    # Ground Truth
    x_real = x_slice.cpu().numpy() * 5.0 + 5.0
    y_gt, _, _, _ = ellipj(x_real, 0.5)
    
    plt.plot(x_slice.cpu(), y_gt, 'k-', label='GT', linewidth=2, alpha=0.5)
    with torch.no_grad():
        y_pred = fused(inputs)
    plt.plot(x_slice.cpu(), y_pred.cpu(), 'r--', label='Fused Pred')
    plt.title("Approximation")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_paper_benchmark_v2()