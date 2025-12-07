#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ==========================================================================
// RESEARCH ENGINEER NOTES:
// 1. __ldg(): Enforced Load Global via Read-Only Cache. Crucial for 'Gather' ops.
// 2. Fixed 'cudaErrorInvalidValue': Removed Shared Memory usage in backward_weights
//    to allow large models (Width=1024, Grid=100) without overflowing cache.
// 3. Learnable Grids: backward_inputs_kernel now computes gradients for min/max.
// ==========================================================================

// ==========================================================================
// MATH HELPERS (Cubic B-Spline)
// ==========================================================================
template <typename T>
__device__ __forceinline__ void compute_cubic_basis(T u, T* b) {
    T u2 = u * u; 
    T u3 = u2 * u;
    // Optimized FMA (Fused Multiply Add) structure where possible by compiler
    b[0] = (1.0f - u3 + 3.0f*u2 - 3.0f*u) / 6.0f;
    b[1] = (3.0f*u3 - 6.0f*u2 + 4.0f) / 6.0f;
    b[2] = (-3.0f*u3 + 3.0f*u2 + 3.0f*u + 1.0f) / 6.0f;
    b[3] = u3 / 6.0f;
}

template <typename T>
__device__ __forceinline__ void compute_cubic_derivative(T u, T* db) {
    T u2 = u * u;
    db[0] = (-3.0f*u2 + 6.0f*u - 3.0f) / 6.0f;
    db[1] = (9.0f*u2 - 12.0f*u) / 6.0f;
    db[2] = (-9.0f*u2 + 6.0f*u + 3.0f) / 6.0f;
    db[3] = 3.0f*u2 / 6.0f; // Simplified: 0.5 * u^2
}

// ==========================================================================
// KERNELS
// ==========================================================================

// 1. Basis Kernel
// Computes the 4 non-zero basis values for every input element.
template <typename T>
__global__ void basis_kernel(
    const T* __restrict__ inputs, 
    T* __restrict__ basis_out, 
    int* __restrict__ index_out, 
    int nbatch, 
    int nfeat, 
    int grid_size, 
    T min, 
    T max) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= nbatch || i >= nfeat) return;
    
    int flat_idx = i * nbatch + b;
    T x = inputs[flat_idx];
    T step = (max - min) / static_cast<T>(grid_size);
    
    // Clamp inputs to grid range to avoid segfaults
    T x_clamped = x;
    if (x_clamped < min) x_clamped = min;
    if (x_clamped > max) x_clamped = max - static_cast<T>(1e-5);
    
    T grid_pos = (x_clamped - min) / step;
    int k = static_cast<int>(floor(grid_pos));
    
    // Boundary check
    if (k < 0) k = 0; 
    if (k > grid_size - 1) k = grid_size - 1;
    
    T u = grid_pos - static_cast<T>(k);
    T b_val[4];
    compute_cubic_basis(u, b_val);
    
    // Memory Layout: [Feature, Batch, 4]
    int out_ptr = (i * nbatch * 4) + (b * 4);
    
    // Unrolled write
    basis_out[out_ptr + 0] = b_val[0]; 
    basis_out[out_ptr + 1] = b_val[1];
    basis_out[out_ptr + 2] = b_val[2]; 
    basis_out[out_ptr + 3] = b_val[3];
    
    index_out[flat_idx] = k;
}

// 2. Forward Kernel
// Fused Gather-GEMM. 
// OPTIMIZATION: Uses __ldg() to load weights through read-only cache.
template <typename T>
__global__ void forward_kernel(
    const T* __restrict__ basis, 
    const int* __restrict__ index, 
    const T* __restrict__ weights, 
    T* __restrict__ output, 
    int nbatch, 
    int nfeat, 
    int nout, 
    int ncoeffs) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= nbatch || o >= nout) return;
    
    T acc = 0.0f;
    
    // Accumulate over input features
    for (int i = 0; i < nfeat; i++) {
        int flat_idx = i * nbatch + b;
        int k = index[flat_idx];
        
        int basis_ptr = (flat_idx * 4);
        int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
        
        // Manual unroll loop size 4
        // Use __ldg to force texture cache read for weights (random access)
        acc += basis[basis_ptr + 0] * __ldg(&weights[w_ptr + 0]);
        acc += basis[basis_ptr + 1] * __ldg(&weights[w_ptr + 1]);
        acc += basis[basis_ptr + 2] * __ldg(&weights[w_ptr + 2]);
        acc += basis[basis_ptr + 3] * __ldg(&weights[w_ptr + 3]);
    }
    
    output[o * nbatch + b] = acc;
}

// 3. Backward Weights (Fixed for Large Width/Grid)
// Removed Shared Memory optimization to prevent overflow on large models.
// Uses direct global atomic adds. 
template <typename T>
__global__ void backward_weights_kernel(
    const T* __restrict__ grad_out, 
    const T* __restrict__ basis, 
    const int* __restrict__ index, 
    T* __restrict__ grad_weights, 
    int nbatch, 
    int nfeat, 
    int nout, 
    int ncoeffs) 
{
    // We parallelize over:
    // Block X: Batch Chunks
    // Block Y: Output Dimensions
    
    int b_start = blockIdx.x * blockDim.x;
    int out_idx = blockIdx.y; // One block per output neuron
    int tid = threadIdx.x;

    if (out_idx >= nout) return;

    // Loop over the batch chunk assigned to this block
    int b_end = min(b_start + blockDim.x, nbatch);
    int b = b_start + tid;
    
    if (b < b_end) {
        // Load Gradient for this batch item and output neuron
        T gy = grad_out[out_idx * nbatch + b];

        // Iterate over ALL input features
        for (int in_idx = 0; in_idx < nfeat; ++in_idx) {
            
            int flat_idx = in_idx * nbatch + b;
            int k = index[flat_idx];
            int basis_ptr = flat_idx * 4;

            // Load Basis
            T b0 = basis[basis_ptr + 0];
            T b1 = basis[basis_ptr + 1];
            T b2 = basis[basis_ptr + 2];
            T b3 = basis[basis_ptr + 3];

            // Calculate Global Weight Pointer
            // Weights: [Out, In, Coeffs]
            int w_base = (out_idx * nfeat * ncoeffs) + (in_idx * ncoeffs) + k;

            // Atomic Add to Global Memory
            atomicAdd(&grad_weights[w_base + 0], gy * b0);
            atomicAdd(&grad_weights[w_base + 1], gy * b1);
            atomicAdd(&grad_weights[w_base + 2], gy * b2);
            atomicAdd(&grad_weights[w_base + 3], gy * b3);
        }
    }
}

// 4. Backward Inputs (With Learnable Grid Gradients)
template <typename T>
__global__ void backward_inputs_kernel(
    const T* __restrict__ grad_out, 
    const T* __restrict__ inputs, 
    const T* __restrict__ weights, 
    T* __restrict__ grad_inputs, 
    T* __restrict__ grad_min, // NEW: Gradient w.r.t grid_min
    T* __restrict__ grad_max, // NEW: Gradient w.r.t grid_max
    int nbatch, 
    int nfeat, 
    int nout, 
    int grid_size, 
    int ncoeffs, 
    T min, 
    T max) 
{
    // Shared memory to reduce atomic contention for min/max gradients
    __shared__ float s_min_grad;
    __shared__ float s_max_grad;
    
    if (threadIdx.x == 0) {
        s_min_grad = 0.0f;
        s_max_grad = 0.0f;
    }
    __syncthreads();

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= nbatch || i >= nfeat) return;
    
    T step = (max - min) / static_cast<T>(grid_size);
    int flat_idx = i * nbatch + b;
    T x = inputs[flat_idx];
    
    // Gradients for inputs outside grid are zero
    if (x < min || x > max) { 
        grad_inputs[flat_idx] = 0.0f; 
        return; 
    }
    
    T x_clamped = x;
    if (x_clamped > max) x_clamped = max - static_cast<T>(1e-5);
    
    T grid_pos = (x_clamped - min) / step;
    int k = static_cast<int>(floor(grid_pos));
    if (k < 0) k = 0; 
    if (k > grid_size - 1) k = grid_size - 1;
    
    T u = grid_pos - static_cast<T>(k);
    T db[4];
    compute_cubic_derivative(u, db); // d(Basis)/du
    
    T acc = 0.0f;
    for (int o = 0; o < nout; o++) {
        T gy = grad_out[o * nbatch + b];
        int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
        
        // Dot product of Weights and dBasis using __ldg
        T dot = db[0] * __ldg(&weights[w_ptr + 0]) + 
                db[1] * __ldg(&weights[w_ptr + 1]) + 
                db[2] * __ldg(&weights[w_ptr + 2]) + 
                db[3] * __ldg(&weights[w_ptr + 3]);
        
        acc += gy * dot;
    }
    
    // 1. Store Input Gradient (dL/dx)
    // Chain Rule: dL/dx = dL/du * du/dx = acc * (1/step)
    T grad_x = acc * (static_cast<T>(1.0f) / step);
    grad_inputs[flat_idx] = grad_x;

    // 2. Compute Min/Max Gradients
    // x_norm = (x - min) / (max - min)
    // dL/dmin = dL/dx * (x_norm - 1)
    // dL/dmax = dL/dx * (-x_norm)
    T x_norm = (x - min) / (max - min);
    T d_min = grad_x * (x_norm - 1.0f);
    T d_max = grad_x * (-x_norm);

    // Atomic add to Block Shared Memory first
    atomicAdd(&s_min_grad, static_cast<float>(d_min));
    atomicAdd(&s_max_grad, static_cast<float>(d_max));

    __syncthreads();

    // Flush Block Shared Memory to Global Memory (Only Thread 0)
    if (threadIdx.x == 0) {
        atomicAdd(grad_min, static_cast<T>(s_min_grad));
        atomicAdd(grad_max, static_cast<T>(s_max_grad));
    }
}

// ==========================================================================
// PYBIND11 DISPATCHERS
// ==========================================================================

std::vector<torch::Tensor> compute_basis(torch::Tensor inputs, int grid_size, double min, double max) {
    int num_features = inputs.size(0); 
    int num_batch = inputs.size(1);
    
    auto basis = torch::empty({num_features, num_batch, 4}, inputs.options());
    auto index = torch::empty({num_features, num_batch}, inputs.options().dtype(torch::kInt32));
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch + 15)/16, (num_features + 15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "basis", ([&] {
        basis_kernel<scalar_t><<<blocks, threads>>>(
            inputs.data_ptr<scalar_t>(), 
            basis.data_ptr<scalar_t>(), 
            index.data_ptr<int>(),
            num_batch, 
            num_features, 
            grid_size, 
            static_cast<scalar_t>(min), 
            static_cast<scalar_t>(max)
        );
    }));
    return {basis, index};
}

torch::Tensor run_forward(torch::Tensor basis, torch::Tensor index, torch::Tensor weights) {
    int num_features = basis.size(0); 
    int num_batch = basis.size(1); 
    int num_outputs = weights.size(0); 
    int num_coeffs = weights.size(2);
    
    auto output = torch::zeros({num_outputs, num_batch}, basis.options());
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_outputs+15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            basis.data_ptr<scalar_t>(), 
            index.data_ptr<int>(), 
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), 
            num_batch, 
            num_features, 
            num_outputs, 
            num_coeffs
        );
    }));
    return output;
}

torch::Tensor run_backward_weights(torch::Tensor grad_out, torch::Tensor basis, torch::Tensor index, int O, int I, int C) {
    int num_batch = grad_out.size(1);
    auto grad_weights = torch::zeros({O, I, C}, grad_out.options());
    
    int threads = 256;
    int chunks = (num_batch + threads - 1) / threads;
    dim3 grid(chunks, O);
    
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "bw_weights", ([&] {
        backward_weights_kernel<scalar_t><<<grid, threads>>>(
            grad_out.data_ptr<scalar_t>(), 
            basis.data_ptr<scalar_t>(), 
            index.data_ptr<int>(),
            grad_weights.data_ptr<scalar_t>(), 
            num_batch, I, O, C
        );
    }));
    return grad_weights;
}

std::vector<torch::Tensor> run_backward_inputs(torch::Tensor grad_out, torch::Tensor inputs, torch::Tensor weights, int grid_size, double min, double max) {
    int num_features = inputs.size(0); 
    int num_batch = inputs.size(1); 
    int num_outputs = weights.size(0); 
    int num_coeffs = weights.size(2);
    
    auto grad_inputs = torch::zeros_like(inputs);
    // NEW: Gradients for grid bounds (scalars)
    auto grad_min = torch::zeros({1}, inputs.options());
    auto grad_max = torch::zeros({1}, inputs.options());
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_features+15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "bw_inputs", ([&] {
        backward_inputs_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), 
            inputs.data_ptr<scalar_t>(), 
            weights.data_ptr<scalar_t>(),
            grad_inputs.data_ptr<scalar_t>(), 
            grad_min.data_ptr<scalar_t>(),
            grad_max.data_ptr<scalar_t>(),
            num_batch, num_features, num_outputs, grid_size, num_coeffs, 
            static_cast<scalar_t>(min), 
            static_cast<scalar_t>(max)
        );
    }));
    // Return list: [grad_inputs, grad_min, grad_max]
    return {grad_inputs, grad_min, grad_max};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_basis", &compute_basis, "Compute Spline Basis");
    m.def("run_forward", &run_forward, "Run Forward Pass");
    m.def("run_backward_weights", &run_backward_weights, "Run Backward Weights");
    m.def("run_backward_inputs", &run_backward_inputs, "Run Backward Inputs");
}