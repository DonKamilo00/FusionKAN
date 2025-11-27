import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os

# Add root to path to find fusion_kan if not installed globally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported.")
except ImportError:
    print("⚠️ FusionKAN not found. Installing...")
    os.system('pip install .')
    from fusion_kan import FusionKANLayer

# ==============================================================================
# 1. DATA GENERATION (Jacobi Elliptic Function)
# ==============================================================================
def generate_data_ellipj(n_samples=50000, device='cuda'):
    # Domain: x in [0, 10], m in [0, 1]
    x = np.random.rand(n_samples) * 10.0
    m = np.random.rand(n_samples)
    sn, _, _, _ = ellipj(x, m)
    
    # Normalize inputs to [-1, 1]
    # x: [0, 10] -> [-1, 1]
    # m: [0, 1]  -> [-1, 1]
    input_data = np.stack([(x - 5.0)/5.0, (m - 0.5)/0.5], axis=1)
    inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
    labels = torch.tensor(sn, dtype=torch.float32).to(device).unsqueeze(1)
    return inputs, labels

# ==============================================================================
# 2. TRAINING LOOP
# ==============================================================================
def train_fusion_kan(steps=2000):
    print("\n🧠 Training FusionKAN on Elliptic Integral...")
    DEVICE = 'cuda'
    
    # Data
    x, y = generate_data_ellipj(50000, DEVICE)
    
    # Model: [2, 2, 1] with Grid=50 (High Precision)
    model = nn.Sequential(
        FusionKANLayer(2, 2, grid_size=50, spline_order=3, grid_range=[-1, 1]),
        FusionKANLayer(2, 1, grid_size=50, spline_order=3, grid_range=[-1, 1], is_output=True)
    ).to(DEVICE)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    
    history = []
    start = time.time()
    
    for i in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        opt.step()
        
        if i % 50 == 0:
            history.append(loss.item())
            scheduler.step()
            
    total_time = time.time() - start
    print(f"  > Finished in {total_time:.2f}s")
    print(f"  > Final MSE: {history[-1]:.6f}")
    return model, history

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
def visualize_results(model, history):
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # A. Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history, label="FusionKAN", color='orange', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title("Training Convergence")
    ax1.set_xlabel("Step (x50)")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # B. 1D Slice (m=0.5)
    ax2 = fig.add_subplot(gs[0, 1])
    x_raw = np.linspace(0, 10, 200)
    x_norm = (x_raw - 5.0) / 5.0
    m_val = 0.5
    m_norm = (m_val - 0.5) / 0.5
    
    inp_np = np.stack([x_norm, np.full_like(x_norm, m_norm)], axis=1)
    inp_torch = torch.tensor(inp_np, dtype=torch.float32, device='cuda')
    
    gt, _, _, _ = ellipj(x_raw, m_val)
    
    with torch.no_grad():
        pred = model(inp_torch).cpu().numpy()
        
    ax2.plot(x_raw, gt, 'k-', label='Ground Truth', linewidth=2, alpha=0.5)
    ax2.plot(x_raw, pred, 'r--', label='FusionKAN Prediction')
    ax2.set_title(f"Function Slice (m={m_val})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # C. 2D Error Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    res = 200
    x_grid = np.linspace(-1, 1, res)
    y_grid = np.linspace(-1, 1, res)
    gx, gy = np.meshgrid(x_grid, y_grid)
    
    flat_inp = np.stack([gx.flatten(), gy.flatten()], axis=1)
    inp_torch = torch.tensor(flat_inp, dtype=torch.float32, device='cuda')
    
    x_real = gx.flatten() * 5.0 + 5.0
    m_real = gy.flatten() * 0.5 + 0.5
    gt_flat, _, _, _ = ellipj(x_real, m_real)
    
    with torch.no_grad():
        pred_flat = model(inp_torch).cpu().numpy().flatten()
        
    error = np.abs(gt_flat - pred_flat).reshape(res, res)
    
    im = ax3.imshow(error, extent=[0, 10, 0, 1], origin='lower', cmap='inferno', aspect='auto')
    ax3.set_title("Absolute Error Heatmap (Full Domain)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("m")
    plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.savefig("paper_benchmark.png")
    print("Saved 'paper_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        model, history = train_fusion_kan()
        visualize_results(model, history)
    else:
        print("CUDA not available.")