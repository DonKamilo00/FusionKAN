--- START OF FILE benchmark_comparison.py ---

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import gc
import sys

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import FusionKAN
try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported successfully.")
except ImportError:
    raise ImportError("Please install fusion_kan first via 'pip install .'")

# Import Original KAN
try:
    from kan.MultKAN import MultKAN
    print("✅ Original KAN code imported successfully.")
except ImportError as e:
    print(f"⚠️ Error importing Original KAN: {e}")
    MultKAN = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {DEVICE}")

def generate_data(n_samples=20000):
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

def train_model(model_name, model, x, y, steps=500, batch_size=512):
    print(f"\n🔄 Training {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    losses = []
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Identify FusionKAN layers for grid updates
    fusion_layers = []
    if "Fusion" in model_name:
        for m in model.modules():
            if isinstance(m, FusionKANLayer):
                fusion_layers.append(m)
    
    for step in range(steps):
        indices = torch.randperm(x.shape[0])[:batch_size]
        batch_x = x[indices]
        batch_y = y[indices]
        
        # --- DYNAMIC GRID UPDATE ---
        # Update grid bounds every 50 steps to adapt to layer activation shifts
        if fusion_layers and step % 50 == 0:
            current_input = batch_x
            for layer in model:
                if isinstance(layer, FusionKANLayer):
                    layer.update_grid(current_input)
                # Propagate input for next layer stats
                # Note: This simple loop assumes Sequential model structure
                current_input = layer(current_input)
        # ---------------------------

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.5f}")

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"✅ {model_name} Finished.")
    print(f"Time: {total_time:.4f}s | Peak VRAM: {peak_mem:.2f} MB | Final Loss: {losses[-1]:.5f}")
    return total_time, peak_mem, losses

def run_benchmark():
    # --- STRESS TEST CONFIG ---
    # Increased width to demonstrate O(N) vs O(G) memory scaling
    WIDTH = [2, 1024, 1] 
    GRID = 100
    K = 3
    STEPS = 200
    BATCH = 8192
    
    print(f"Benchmark Config: Width={WIDTH}, Grid={GRID}, Batch={BATCH}")
    
    x, y = generate_data(n_samples=50000)
    results = {}
    
    # 1. Train Original KAN
    if MultKAN is not None:
        try:
            print("Initializing Original KAN...")
            # symbolic_enabled=False to compare numerical engine only
            orig_model = MultKAN(width=WIDTH, grid=GRID, k=K, symbolic_enabled=False, device=DEVICE)
            
            # Reset peak memory before tracking
            torch.cuda.reset_peak_memory_stats()
            t_orig, m_orig, l_orig = train_model("Original KAN", orig_model, x, y, steps=STEPS, batch_size=BATCH)
            
            if t_orig > 0:
                results['Original KAN'] = {'time': t_orig, 'mem': m_orig, 'loss': l_orig}
            
            del orig_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Original KAN failed (likely OOM): {e}")
            # If OOM, record null results to show failure in plot
            results['Original KAN'] = {'time': 0, 'mem': 0, 'loss': []}

    # 2. Train FusionKAN
    print("Initializing FusionKAN...")
    fused_model = nn.Sequential(
        FusionKANLayer(2, 1024, grid_size=GRID, spline_order=K, use_node_activation=False),
        FusionKANLayer(1024, 1, grid_size=GRID, spline_order=K, is_output=True)
    ).to(DEVICE)
    
    # Warmup kernel compilation
    _ = fused_model(x[:10])
    
    torch.cuda.reset_peak_memory_stats()
    t_fused, m_fused, l_fused = train_model("FusionKAN", fused_model, x, y, steps=STEPS, batch_size=BATCH)
    results['FusionKAN'] = {'time': t_fused, 'mem': m_fused, 'loss': l_fused}

    plot_results(results)

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    
    # Handle cases where Original KAN might have failed
    times = []
    mems = []
    valid_names = []
    
    for n in names:
        if results[n]['time'] > 0:
            times.append(results[n]['time'])
            mems.append(results[n]['mem'])
            valid_names.append(n)
        else:
            print(f"Skipping {n} in bar charts due to failure.")
            
    # Time
    ax1.bar(valid_names, times, color=['#FF5733', '#33FF57'])
    ax1.set_title('Training Time (Lower is Better)')
    ax1.set_ylabel('Seconds')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}s", ha='center', va='bottom')

    # Memory
    ax2.bar(valid_names, mems, color=['#FF5733', '#33FF57'])
    ax2.set_title('Peak VRAM (Lower is Better)')
    ax2.set_ylabel('MB')
    for i, v in enumerate(mems):
        ax2.text(i, v, f"{v:.0f} MB", ha='center', va='bottom')

    # Loss
    for name in names:
        if len(results[name]['loss']) > 0:
            ax3.plot(results[name]['loss'], label=name)
            
    ax3.set_title('Convergence (MSE Loss)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\n📊 Saved to benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    run_benchmark()