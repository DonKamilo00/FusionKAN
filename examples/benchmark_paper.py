import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os

sys.path.append(os.getcwd())
from fusion_kan import FusionKANLayer
try:
    from kan.MultKAN import MultKAN
except:
    MultKAN = None

def run_aligned_benchmark():
    print("--- 🧪 Aligned KAN Benchmark (Equal Grid Size) ---")
    DEVICE = 'cuda'
    
    # 1. Data
    def generate_data(n=50000):
        x = np.random.rand(n) * 10.0
        m = np.random.rand(n)
        sn, _, _, _ = ellipj(x, m)
        inputs = torch.tensor(np.stack([(x-5)/5, (m-0.5)/0.5], 1), dtype=torch.float32).to(DEVICE)
        labels = torch.tensor(sn, dtype=torch.float32).to(DEVICE).unsqueeze(1)
        return inputs, labels

    train_x, train_y = generate_data(50000)
    
    # 2. Config (EXACT MATCH)
    GRID = 20          # Match Original
    WIDTH = [2, 2, 1]
    RANGE = [-1.2, 1.2] 
    
    # 3. Training Loop with Regularization
    def train(name, model, steps=2000):
        print(f"Training {name}...")
        # Weight decay = 1e-4 acts as smoothness regularization
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) 
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.995)
        
        history = []
        start = time.time()
        for i in range(steps):
            opt.zero_grad()
            loss = nn.MSELoss()(model(train_x), train_y)
            loss.backward()
            opt.step()
            if i % 50 == 0: 
                history.append(loss.item())
                sch.step()
        
        print(f"  > Time: {time.time()-start:.2f}s | Final MSE: {history[-1]:.5f}")
        return history

    # 4. Models
    results = {}
    
    if MultKAN:
        torch.cuda.empty_cache()
        orig = MultKAN(width=WIDTH, grid=GRID, k=3, symbolic_enabled=False, device=DEVICE)
        results['Original'] = train("Original", orig)

    torch.cuda.empty_cache()
    fused = nn.Sequential(
        FusionKANLayer(2, 2, grid_size=GRID, grid_range=RANGE), # Grid=20
        FusionKANLayer(2, 1, grid_size=GRID, grid_range=RANGE, is_output=True)
    ).to(DEVICE)
    results['Fused'] = train("Fused", fused)
    
    # 5. Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        plt.plot(hist, label=name)
    plt.yscale('log')
    plt.title("Convergence (Equal Grid=20)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x_raw = np.linspace(0, 10, 200)
    inp = torch.tensor(np.stack([(x_raw-5)/5, np.zeros_like(x_raw)], 1), dtype=torch.float32).to(DEVICE)
    gt, _, _, _ = ellipj(x_raw, 0.5)
    
    with torch.no_grad():
        pred = fused(inp).cpu().numpy()
        
    plt.plot(x_raw, gt, 'k', label='GT', alpha=0.5, linewidth=2)
    plt.plot(x_raw, pred, 'r--', label='Fused Pred')
    plt.title(f"Fused Fit (m=0.5)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_aligned_benchmark()