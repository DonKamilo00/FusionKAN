
import torch
import time
import sys
import os

# Add current dir to path so we can import fusion_kan if not installed globally
sys.path.append(os.getcwd())

try:
    from fusion_kan import FusionKANLayer
except ImportError:
    # Fallback if running from root without install
    import sys
    sys.path.append('..')
    from fusion_kan import FusionKANLayer

def run():
    if not torch.cuda.is_available():
        print("No GPU found!")
        return

    print("--- FusionKAN Health Check ---")

    # 1. Setup
    BATCH = 100000
    DIM = 64
    GRID = 32
    
    print(f"Initializing Layer [In={DIM}, Out={DIM}, Grid={GRID}]...")
    layer = FusionKANLayer(DIM, DIM, grid_size=GRID).cuda()
    x = torch.randn(BATCH, DIM).cuda()
    
    print(f"Input Shape: {x.shape}")

    # 2. Warmup (Compiles kernels if JIT, warms up allocator)
    print("Warming up...")
    y = layer(x)
    loss = y.sum()
    loss.backward()
    torch.cuda.synchronize()

    # 3. Benchmark Loop
    print("Running Benchmark (100 iterations)...")
    start = time.time()
    
    for _ in range(100):
        y = layer(x)
        loss = y.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    iter_time = (total_time / 100) * 1000 # ms
    
    print(f"\n✅ SUCCESS!")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Time per Iteration: {iter_time:.2f} ms")
    
    # 4. Gradient Check
    grad_norm = layer.spline_weight.grad.norm().item()
    print(f"Gradient Norm: {grad_norm:.4f} (Should be non-zero)")

if __name__ == "__main__":
    run()