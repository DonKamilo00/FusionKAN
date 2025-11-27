import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import gc
from collections import defaultdict

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from fusion_kan import FusionKANLayer
    print("✅ FusionKAN imported.")
except ImportError:
    print("❌ FusionKAN not found. Install via 'pip install .'")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- THE METRICS COLLECTOR ---
class DeepDiveTracker:
    def __init__(self):
        self.stats = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        # Register hooks for every FusionKANLayer
        for name, module in model.named_modules():
            if isinstance(module, FusionKANLayer):
                # Forward Hook: Check Inputs
                self.hooks.append(module.register_forward_hook(self.get_activation_hook(name)))
                # Backward Hook: Check Gradients
                self.hooks.append(module.register_full_backward_hook(self.get_gradient_hook(name)))

    def get_activation_hook(self, name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            
            # 1. Range Check
            curr_min = x.min().item()
            curr_max = x.max().item()
            
            # 2. Grid Bounds Check (Vital for FusionKAN)
            grid_min = module.grid_min
            grid_max = module.grid_max
            
            # % of data outside grid
            out_of_bounds = ((x < grid_min) | (x > grid_max)).float().mean().item() * 100
            
            self.stats[f"{name}_act_mean"].append(x.mean().item())
            self.stats[f"{name}_act_std"].append(x.std().item())
            self.stats[f"{name}_act_min"].append(curr_min)
            self.stats[f"{name}_act_max"].append(curr_max)
            self.stats[f"{name}_oob_pct"].append(out_of_bounds)
        return hook

    def get_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_input is a tuple (grad_x, grad_weights...)
            if grad_input[0] is not None:
                g = grad_input[0].detach()
                self.stats[f"{name}_grad_norm"].append(g.norm().item())
                self.stats[f"{name}_grad_mean"].append(g.abs().mean().item())
        return hook

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.stats = defaultdict(list)

# --- DATA GENERATION ---
def generate_data(n_samples=5000):
    x = torch.rand(n_samples, 2).to(DEVICE) * 2 - 1
    # Function: exp(sin(pi*x) + y^2)
    y = torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)
    return x, y.unsqueeze(1)

# --- RUNNER ---
def run_stress_test():
    tracker = DeepDiveTracker()
    
    # Setup Model
    # NOTE: Ensure you are using the FIXED grid_range=[-3, 3] in layer.py
    model = nn.Sequential(
        FusionKANLayer(2, 32, grid_size=10, spline_order=3, grid_range=[-3, 3]), 
        FusionKANLayer(32, 1, grid_size=10, spline_order=3, grid_range=[-3, 3], is_output=True)
    ).to(DEVICE)
    
    tracker.register_hooks(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    x, y = generate_data()
    
    print("\n🕵️ STARTING DEEP DIVE...")
    print(f"Model Configuration: Grid Range [-3, 3] expected.")
    
    epochs = 300
    loss_history = []
    
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if i % 50 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

    # --- PLOTTING ---
    plot_deep_dive(loss_history, tracker.stats)
    tracker.clear()

def plot_deep_dive(losses, stats):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(losses)
    ax1.set_yscale('log')
    ax1.set_title("Loss Curve (Log Scale)")
    ax1.set_xlabel("Steps")
    ax1.grid(True, alpha=0.3)

    # 2. Activations (Layer 0)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(stats['0_act_min'], label='Min', linestyle='--', alpha=0.5)
    ax2.plot(stats['0_act_max'], label='Max', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(stats['0_act_mean'])), 
                     np.array(stats['0_act_mean']) - np.array(stats['0_act_std']),
                     np.array(stats['0_act_mean']) + np.array(stats['0_act_std']), 
                     alpha=0.3, label='Mean ± Std')
    
    # Draw Grid Boundaries
    ax2.axhline(y=-3, color='r', linestyle=':', label='Grid Bound')
    ax2.axhline(y=3, color='r', linestyle=':')
    
    ax2.set_title("Layer 0 Activations vs Grid")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Out of Bounds Percentage (CRITICAL)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(stats['0_oob_pct'], label='Layer 0 % OOB', color='red')
    if '1_oob_pct' in stats:
        ax3.plot(stats['1_oob_pct'], label='Layer 1 % OOB', color='orange')
    ax3.set_title("Percentage of Dead Inputs (Out of Grid)")
    ax3.set_ylabel("%")
    ax3.set_ylim(-1, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gradient Norms
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(stats['0_grad_norm'], label='Layer 0 Grad Norm')
    if '1_grad_norm' in stats:
        ax4.plot(stats['1_grad_norm'], label='Layer 1 Grad Norm')
    ax4.set_title("Gradient Norms (Check for explosions)")
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('deep_dive_analysis.png')
    print("\n📊 Saved analysis to 'deep_dive_analysis.png'")
    print("Check the 'Dead Inputs' graph. If it is > 0%, inputs are clipping.")
    plt.show()

if __name__ == "__main__":
    run_stress_test()