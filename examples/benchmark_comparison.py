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
    
    for step in range(steps):
        indices = torch.randperm(x.shape[0])[:batch_size]
        batch_x = x[indices]
        batch_y = y[indices]
        
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
    # --- STRATEGY TO BEAT ORIGINAL KAN ---
    WIDTH = [2, 64, 1]
    
    # GRID=30 with range [-3,3] gives similar resolution density 
    # as GRID=10 with range [-1,1].
    # This recovers the accuracy we lost by expanding the range.
    GRID = 30 
    K = 3
    STEPS = 500
    BATCH = 2048
    
    x, y = generate_data(n_samples=20000)
    results = {}
    
    # 1. Train Original KAN
    if MultKAN is not None:
        try:
            # Original KAN defaults to grid=10 usually
            orig_model = MultKAN(width=WIDTH, grid=10, k=K, symbolic_enabled=False, device=DEVICE)
            t_orig, m_orig, l_orig = train_model("Original KAN", orig_model, x, y, steps=STEPS, batch_size=BATCH)
            if t_orig > 0:
                results['Original KAN'] = {'time': t_orig, 'mem': m_orig, 'loss': l_orig}
            del orig_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Original KAN failed: {e}")

    # 2. Train FusionKAN (Pure Mode)
    fused_model = nn.Sequential(
        # use_node_activation=False removes LayerNorm/PReLU to match Original KAN architecture
        FusionKANLayer(2, 64, grid_size=GRID, spline_order=K, use_node_activation=False),
        FusionKANLayer(64, 1, grid_size=GRID, spline_order=K, is_output=True)
    ).to(DEVICE)
    
    # Warmup kernel
    _ = fused_model(x[:10])
    
    t_fused, m_fused, l_fused = train_model("FusionKAN", fused_model, x, y, steps=STEPS, batch_size=BATCH)
    results['FusionKAN'] = {'time': t_fused, 'mem': m_fused, 'loss': l_fused}

    plot_results(results)

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    mems = [results[n]['mem'] for n in names]
    
    # Time
    ax1.bar(names, times, color=['#FF5733', '#33FF57'])
    ax1.set_title('Training Time (Lower is Better)')
    ax1.set_ylabel('Seconds')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}s", ha='center', va='bottom')
    if len(names) > 1:
        speedup = results['Original KAN']['time'] / results['FusionKAN']['time']
        ax1.text(0.5, max(times)*0.5, f"{speedup:.1f}x Speedup", ha='center', bbox=dict(facecolor='white'))

    # Memory
    ax2.bar(names, mems, color=['#FF5733', '#33FF57'])
    ax2.set_title('Peak VRAM (Lower is Better)')
    ax2.set_ylabel('MB')
    for i, v in enumerate(mems):
        ax2.text(i, v, f"{v:.0f} MB", ha='center', va='bottom')

    # Loss
    for name in names:
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