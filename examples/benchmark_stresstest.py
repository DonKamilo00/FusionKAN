import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import gc
import sys
import os

# Fix path
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

# Imports
from fusion_kan import FusionKANLayer
from kan.MultKAN import MultKAN

def run_stress_test():
    print("\n--- 🏋️ FUSION-KAN STRESS TEST (Batch=65k) ---")
    
    DEVICE = 'cuda'
    BATCH = 65536 # Huge batch
    STEPS = 100   # Short run to measure speed
    WIDTH = [2, 64, 1]
    GRID = 10
    K = 3
    
    # Data
    x = torch.rand(BATCH, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2).unsqueeze(1)
    
    results = {}
    
    # 1. Original
    print("Testing Original KAN...")
    torch.cuda.empty_cache()
    try:
        model = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False, device=DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(STEPS):
            opt.zero_grad()
            loss = torch.nn.MSELoss()(model(x), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        results['Original'] = time.time() - start
        print(f"Original Time: {results['Original']:.4f}s")
    except RuntimeError as e:
        print(f"Original KAN Crashed: {e}")
        results['Original'] = None

    # 2. FusionKAN
    print("Testing FusionKAN...")
    torch.cuda.empty_cache()
    model = torch.nn.Sequential(
        FusionKANLayer(2, 64, grid_size=GRID, spline_order=K),
        FusionKANLayer(64, 1, grid_size=GRID, spline_order=K, is_output=True)
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = torch.nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    results['Fusion'] = time.time() - start
    print(f"Fusion Time: {results['Fusion']:.4f}s")

    # 3. Print Stats
    if results['Original']:
        print(f"\n🔥 SPEEDUP: {results['Original'] / results['Fusion']:.2f}x")
    else:
        print("\n🔥 SPEEDUP: Infinite (Original Crashed)")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_stress_test()