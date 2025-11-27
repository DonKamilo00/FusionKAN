import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time
import sys
import os

# ==============================================================================
# 1. SETUP & IMPORTS
# ==============================================================================
# Add current directory to path to find 'kan' folder
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

# Import FusionKAN
try:
    from fusion_kan import FusionKANLayer
except ImportError:
    # Fallback if running inside the repo structure without pip install
    if os.path.exists("fusion_kan"):
        from fusion_kan.layer import FusionKANLayer
    else:
        raise ImportError("FusionKAN not found. Run 'pip install .'")

# Import Original KAN
try:
    from kan.MultKAN import MultKAN
except ImportError:
    # Try looking one level up or in current dir
    try:
        sys.path.append(os.path.join(os.getcwd(), 'kan'))
        from MultKAN import MultKAN
    except ImportError:
        print("⚠️ Original KAN code not found. Placeholder will be used.")
        MultKAN = None

# ==============================================================================
# 2. DATA GENERATION
# ==============================================================================
def generate_data_ellipj(n_samples=50000, device='cuda'):
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
# 3. TRAINING LOOP
# ==============================================================================
def train_model_fast(model, x, y, steps=2000):
    # Use Adam for fair speed comparison (LBFGS is too slow per step)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    
    losses = []
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(steps):
        opt.zero_grad()
        # Handle Original KAN specific call signature vs Standard nn.Module
        try:
            pred = model(x)
        except:
            pred = model(x)
            
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        opt.step()
        
        if i % 50 == 0:
            losses.append(loss.item())
            scheduler.step()
    
    torch.cuda.synchronize()        
    return losses, time.time() - start

# ==============================================================================
# 4. VISUALIZATION DASHBOARD
# ==============================================================================
def visualize_paper_results(orig_model, fused_model, orig_stats, fused_stats):
    plt.style.use('default')
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # --- A. Convergence ---
    ax1 = fig.add_subplot(gs[0, :])
    if orig_stats:
        ax1.plot(orig_stats['loss'], label=f"Original ({orig_stats['time']:.1f}s)")
    ax1.plot(fused_stats['loss'], label=f"Fused ({fused_stats['time']:.1f}s)", linestyle='--', color='orange', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title("Training Convergence (Elliptic Integral)")
    ax1.set_xlabel("Step (x50)")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- B. 1D Slices ---
    N_TEST = 200
    x_raw = np.linspace(0, 10, N_TEST)
    x_norm = (x_raw - 5.0) / 5.0
    
    m_vals = [0.1, 0.5, 0.9]
    axes_slices = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    for i, m_val in enumerate(m_vals):
        m_norm = (m_val - 0.5) / 0.5
        inp_np = np.stack([x_norm, np.full_like(x_norm, m_norm)], axis=1)
        inp_torch = torch.tensor(inp_np, dtype=torch.float32, device='cuda')
        
        # Ground Truth
        gt, _, _, _ = ellipj(x_raw, m_val)
        
        with torch.no_grad():
            pred_fused = fused_model(inp_torch).cpu().numpy()
            if orig_model:
                pred_orig = orig_model(inp_torch).cpu().numpy()
            
        ax = axes_slices[i]
        ax.plot(x_raw, gt, 'k-', label='GT', linewidth=2, alpha=0.3)
        if orig_model:
            ax.plot(x_raw, pred_orig, 'b:', label='Original')
        ax.plot(x_raw, pred_fused, 'r--', label='Fused')
        ax.set_title(f"m={m_val}")
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)

    # --- C. 2D Error Heatmap (Fused) ---
    res = 100
    x_grid = np.linspace(-1, 1, res)
    y_grid = np.linspace(-1, 1, res)
    gx, gy = np.meshgrid(x_grid, y_grid)
    
    flat_inp = np.stack([gx.flatten(), gy.flatten()], axis=1)
    inp_torch = torch.tensor(flat_inp, dtype=torch.float32, device='cuda')
    
    x_real = gx.flatten() * 5.0 + 5.0
    m_real = gy.flatten() * 0.5 + 0.5
    gt_flat, _, _, _ = ellipj(x_real, m_real)
    
    with torch.no_grad():
        pred_flat = fused_model(inp_torch).cpu().numpy().flatten()
        
    error = np.abs(gt_flat - pred_flat).reshape(res, res)
    
    ax_heat = fig.add_subplot(gs[2, 0])
    im = ax_heat.imshow(error, extent=[0, 10, 0, 1], origin='lower', cmap='inferno')
    ax_heat.set_title("Fused KAN Error Heatmap")
    ax_heat.set_xlabel("x")
    ax_heat.set_ylabel("m")
    plt.colorbar(im, ax=ax_heat)

    # --- D. 3D Surface ---
    ax_3d = fig.add_subplot(gs[2, 1:], projection='3d')
    sub = 4
    ax_3d.plot_surface(
        gx.reshape(res, res)[::sub, ::sub] * 5 + 5, 
        gy.reshape(res, res)[::sub, ::sub] * 0.5 + 0.5, 
        pred_flat.reshape(res, res)[::sub, ::sub], 
        cmap='viridis', edgecolor='none', alpha=0.9
    )
    ax_3d.set_title("Learned Manifold")
    
    plt.tight_layout()
    plt.savefig("kan_dashboard.png")
    print("Saved 'kan_dashboard.png'")
    plt.show()

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("--- 🏃 Running Visualization Benchmark ---")
        x, y = generate_data_ellipj(50000)
        
        # 1. Train Original (if available)
        orig_stats = None
        orig = None
        if MultKAN is not None:
            print("Training Original KAN...")
            try:
                # Standard Paper Config: [2, 2, 1], Grid=20
                orig = MultKAN(width=[2,2,1], grid=20, k=3, symbolic_enabled=False, device='cuda')
                loss_orig, time_orig = train_model_fast(orig, x, y)
                orig_stats = {'loss': loss_orig, 'time': time_orig}
            except Exception as e:
                print(f"Original KAN failed: {e}")

        # 2. Train Fused
        print("Training Fused KAN...")
        # Use Grid=50 for high precision fitting of special functions
        fused = nn.Sequential(
            FusionKANLayer(2, 2, grid_size=50, spline_order=3, grid_range=[-1,1]),
            FusionKANLayer(2, 1, grid_size=50, spline_order=3, grid_range=[-1,1], is_output=True)
        ).cuda()
        loss_fused, time_fused = train_model_fast(fused, x, y)
        
        # 3. Visualize
        visualize_paper_results(
            orig, fused,
            orig_stats,
            {'loss': loss_fused, 'time': time_fused}
        )