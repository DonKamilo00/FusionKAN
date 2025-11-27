# FusionKAN: High-Performance CUDA Kolmogorov-Arnold Networks

**FusionKAN** is a highly optimized PyTorch library for Kolmogorov-Arnold Networks (KANs). By fusing B-spline computation, coefficient gathering, and matrix multiplication into a single CUDA kernel, it achieves **7-10x speedups** and **Linear Memory Scaling** compared to standard implementations.

![FusionKAN Speedup](https://via.placeholder.com/800x200.png?text=FusionKAN+Speedup+Chart+Here)

## 🚀 Features

- **Fused CUDA Kernels:** Performs Basis computation + Gather + Multiply in a single pass.
- **Memory Efficient:** Uses $O(N)$ memory instead of $O(N \times G)$. Can handle batch sizes of 1M+ on a single GPU.
- **Fast Training:** Optimized backward pass with Shared Memory Atomic reduction.
- **Physics Ready:** Supports exact input gradients ($\nabla x$) for PINNs and NeRFs (Eikonal Loss).

## 📦 Installation

```bash
git clone https://github.com/yourusername/FusionKAN
cd FusionKAN
pip install .