import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os
import math

# Ensure we can import fusion_kan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported.")
except ImportError:
    print("⚠️ FusionKAN not installed. Installing...")
    os.system('pip install .')
    from fusion_kan import FusionKANLayer

try:
    from kan.MultKAN import MultKAN
except ImportError:
    MultKAN = None

# ==============================================================================
# 1. DATA GENERATION
# ==============================================================================
def generate_data_ellipj(n_samples=50000, device='cuda'):
    # Task: Jacobi Elliptic Function sn(x, m)
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
# 2. TRAINING UTILS
# ==============================================================================
def train_loop(name, model, x, y, steps=2000):
    print(f"\n🧠 Training {name}...")
    # Lower LR is safer for splines
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.995)
    
    history = []
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(steps):
        opt.zero_grad()
        try:
            pred = model(x)
        except:
            pred = model(x) # Handle Original KAN quirks
            
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        
        if i % 50 == 0: 
            history.append(loss.item())
            sch.step()
            
    torch.cuda.synchronize()
    total_time = time.time() - start
    print(f"  > Finished in {total_time:.2f}s | Final MSE: {history[-1]:.5f}")
    return history, total_time

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def run_paper_benchmark():
    print("--- 🧪 KAN Paper Benchmark: Elliptic Integral ---")
    DEVICE = 'cuda'
    
    # 1. Data
    train_x, train_y = generate_data_ellipj(50000, DEVICE)
    
    # 2. Config
    GRID = 50
    RANGE = [-1.2, 1.2] # Padded range prevents boundary issues
    WIDTH = [2, 2, 1]
    
    # 3. Models
    orig_res = None
    if MultKAN:
        try:
            print("Initializing Original KAN...")
            model_orig = MultKAN(width=WIDTH, grid=20, k=3, symbolic_enabled=False, device=DEVICE)
            h_orig, t_orig = train_loop("Original", model_orig, train_x, train_y)
            orig_res = {'loss': h_orig, 'time': t_orig}
        except Exception as e:
            print(f"Original KAN failed: {e}")

    print("Initializing FusionKAN...")
    model_fused = nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, grid_range=RANGE),
        FusionKANLayer(2, 1, grid_size=GRID, grid_range=RANGE, is_output=True)
    ).to(DEVICE)
    
    h_fused, t_fused = train_loop("Fused", model_fused, train_x, train_y)
    
    # 4. Visualize
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 5))
    
    # Plot Convergence
    ax1 = fig.add_subplot(1, 2, 1)
    if orig_res:
        ax1.plot(orig_res['loss'], label=f"Original ({orig_res['time']:.1f}s)")
    ax1.plot(h_fused, label=f"Fused ({t_fused:.1f}s)", linestyle='--', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title("Convergence (MSE)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Slice Fit (m=0.5)
    ax2 = fig.add_subplot(1, 2, 2)
    x_raw = np.linspace(0, 10, 200)
    x_norm = (x_raw - 5.0) / 5.0
    m_val = 0.5
    m_norm = (m_val - 0.5) / 0.5 
    
    inp = torch.tensor(np.stack([x_norm, np.full_like(x_norm, m_norm)], 1), dtype=torch.float32).to(DEVICE)
    
    # Ground Truth
    gt, _, _, _ = ellipj(x_raw, m_val)
    
    # Pred
    with torch.no_grad():
        pred = model_fused(inp).cpu().numpy()
        
    ax2.plot(x_raw, gt, 'k', label='Ground Truth', alpha=0.6, linewidth=2)
    ax2.plot(x_raw, pred, 'r--', label='Fused Prediction')
    ax2.set_title("Function Fit (m=0.5)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("paper_benchmark_results.png")
    print("Saved plot to 'paper_benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_paper_benchmark()
    else:
        print("No GPU detected.")