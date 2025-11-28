#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ==========================================================================
// RESEARCH ENGINEER NOTES:
// 1. __ldg(): Enforced Load Global via Read-Only Cache. Crucial for 'Gather' ops.
// 2. atomicAdd: Fixed the missing write-back in backward pass.
// 3. __forceinline__: Ensure math helpers don't add function call overhead.
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
// Complexity: O(N)
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

// 3. Backward Weights
// Computes dL/dW. 
// OPTIMIZATION: Uses Shared Memory reduction to minimize Global Atomics.
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
    // Shared memory for this block's accumulation
    extern __shared__ float s_accum[];
    
    int i = blockIdx.y;   // Feature index
    int chunk = blockIdx.x; // Batch chunk index
    int tid = threadIdx.x;
    
    // 1. Initialize shared memory to 0
    // Each block handles specific Feature 'i'. Shared mem size = [nout * ncoeffs]
    for (int t = tid; t < nout * ncoeffs; t += blockDim.x) {
        s_accum[t] = 0.0f;
    }
    __syncthreads();
    
    // 2. Accumulate gradients over the batch chunk
    int stride = blockDim.x * gridDim.x; // Global stride if needed, but here we process specific chunks
    // Actually, logical logic: chunks split nbatch.
    
    int start_b = chunk * blockDim.x;
    int end_b = min(start_b + blockDim.x, nbatch);
    
    // Loop over this block's assigned batch elements
    // Note: To optimize, we loop stride based on blockDim inside the chunk
    for (int b_offset = tid; start_b + b_offset < end_b; b_offset += blockDim.x) {
        int b = start_b + b_offset;
        
        int flat_idx = i * nbatch + b;
        int basis_ptr = flat_idx * 4;
        
        // Load basis for this input
        T b0 = basis[basis_ptr + 0]; 
        T b1 = basis[basis_ptr + 1]; 
        T b2 = basis[basis_ptr + 2]; 
        T b3 = basis[basis_ptr + 3];
        
        int k = index[flat_idx]; // Which grid coefficient
        
        // For every output dimension
        for (int o = 0; o < nout; o++) {
            T gy = grad_out[o * nbatch + b];
            
            // Shared memory pointer: Row 'o', offset 'k'
            int s_ptr = (o * ncoeffs) + k;
            
            // Atomic Add to Shared Memory
            // (Much faster than global atomic add)
            atomicAdd(&s_accum[s_ptr + 0], static_cast<float>(gy * b0));
            atomicAdd(&s_accum[s_ptr + 1], static_cast<float>(gy * b1));
            atomicAdd(&s_accum[s_ptr + 2], static_cast<float>(gy * b2));
            atomicAdd(&s_accum[s_ptr + 3], static_cast<float>(gy * b3));
        }
    }
    __syncthreads();
    
    // 3. FLUSH TO GLOBAL MEMORY (The Critical Fix)
    for (int t = tid; t < nout * ncoeffs; t += blockDim.x) {
        int o = t / ncoeffs; 
        int c = t % ncoeffs;
        
        // Global pointer: [Output, Feature, Coeff]
        int global_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + c;
        
        // Atomic Add to Global Memory
        // Required because multiple blocks (chunks) handle the same (Feature, Output)
        atomicAdd(&grad_weights[global_ptr], static_cast<T>(s_accum[t]));
    }
}

// 4. Backward Inputs
// Computes dL/dX.
template <typename T>
__global__ void backward_inputs_kernel(
    const T* __restrict__ grad_out, 
    const T* __restrict__ inputs, 
    const T* __restrict__ weights, 
    T* __restrict__ grad_inputs, 
    int nbatch, 
    int nfeat, 
    int nout, 
    int grid_size, 
    int ncoeffs, 
    T min, 
    T max) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= nbatch || i >= nfeat) return;
    
    T step = (max - min) / static_cast<T>(grid_size);
    int flat_idx = i * nbatch + b;
    T x = inputs[flat_idx];
    
    // If input is out of bounds, gradient is 0
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
        
        // Dot product of Weights and dBasis
        T dot = db[0] * __ldg(&weights[w_ptr + 0]) + 
                db[1] * __ldg(&weights[w_ptr + 1]) + 
                db[2] * __ldg(&weights[w_ptr + 2]) + 
                db[3] * __ldg(&weights[w_ptr + 3]);
        
        acc += gy * dot;
    }
    
    // Chain Rule: dL/dx = dL/du * du/dx
    // du/dx = 1/step
    grad_inputs[flat_idx] = acc * (static_cast<T>(1.0f) / step);
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
    
    // Initialize Gradient to 0
    auto grad_weights = torch::zeros({O, I, C}, grad_out.options());
    
    // We launch:
    // Grid Y: Input Features (I)
    // Grid X: Batch Chunks (nbatch / 256)
    int threads = 256;
    int chunks = (num_batch + threads - 1) / threads;
    
    dim3 grid(chunks, I);
    
    // Shared Memory Size: Outputs * Coeffs * sizeof(float)
    // Careful: If O*C is large, this might exceed shared mem limits. 
    // FusionKAN usually has small O (64-128) and C (8-10).
    int shared_mem = O * C * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "bw_weights", ([&] {
        backward_weights_kernel<scalar_t><<<grid, threads, shared_mem>>>(
            grad_out.data_ptr<scalar_t>(), 
            basis.data_ptr<scalar_t>(), 
            index.data_ptr<int>(),
            grad_weights.data_ptr<scalar_t>(), 
            num_batch, I, O, C
        );
    }));
    return grad_weights;
}

torch::Tensor run_backward_inputs(torch::Tensor grad_out, torch::Tensor inputs, torch::Tensor weights, int grid_size, double min, double max) {
    int num_features = inputs.size(0); 
    int num_batch = inputs.size(1); 
    int num_outputs = weights.size(0); 
    int num_coeffs = weights.size(2);
    
    auto grad_inputs = torch::zeros_like(inputs);
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_features+15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "bw_inputs", ([&] {
        backward_inputs_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), 
            inputs.data_ptr<scalar_t>(), 
            weights.data_ptr<scalar_t>(),
            grad_inputs.data_ptr<scalar_t>(), 
            num_batch, 
            num_features, 
            num_outputs, 
            grid_size, 
            num_coeffs, 
            static_cast<scalar_t>(min), 
            static_cast<scalar_t>(max)
        );
    }));
    return grad_inputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_basis", &compute_basis, "Compute Spline Basis");
    m.def("run_forward", &run_forward, "Run Forward Pass");
    m.def("run_backward_weights", &run_backward_weights, "Run Backward Weights");
    m.def("run_backward_inputs", &run_backward_inputs, "Run Backward Inputs");
}