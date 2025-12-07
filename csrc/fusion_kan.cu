#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// ==========================================================================
// MIXED PRECISION SUPPORT (FP16)
// 1. Math is done in float32 for stability.
// 2. Storage is scalar_t (float or half).
// 3. Custom atomicAdd wrapper for Half precision.
// ==========================================================================

// Helper: Atomic Add that handles __half conversion
template <typename T>
__device__ __forceinline__ void fast_atomic_add(T* address, T val) {
    atomicAdd(address, val);
}

// Specialization for at::Half (maps to __half in CUDA)
template <>
__device__ __forceinline__ void fast_atomic_add(at::Half* address, at::Half val) {
    // T4 (CC 7.5) supports atomicAdd for __half natively
    #if __CUDA_ARCH__ >= 600
        atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
    #else
        // Fallback for very old GPUs (not T4): CAS loop (omitted for brevity)
    #endif
}

template <typename T>
__device__ __forceinline__ void compute_cubic_basis(float u, float* b) {
    float u2 = u * u; 
    float u3 = u2 * u;
    b[0] = (1.0f - u3 + 3.0f*u2 - 3.0f*u) / 6.0f;
    b[1] = (3.0f*u3 - 6.0f*u2 + 4.0f) / 6.0f;
    b[2] = (-3.0f*u3 + 3.0f*u2 + 3.0f*u + 1.0f) / 6.0f;
    b[3] = u3 / 6.0f;
}

template <typename T>
__device__ __forceinline__ void compute_cubic_derivative(float u, float* db) {
    float u2 = u * u;
    db[0] = (-3.0f*u2 + 6.0f*u - 3.0f) / 6.0f;
    db[1] = (9.0f*u2 - 12.0f*u) / 6.0f;
    db[2] = (-9.0f*u2 + 6.0f*u + 3.0f) / 6.0f;
    db[3] = 3.0f*u2 / 6.0f; 
}

// 1. Basis Kernel
template <typename T>
__global__ void basis_kernel(
    const T* __restrict__ inputs, 
    T* __restrict__ basis_out, 
    int* __restrict__ index_out, 
    int nbatch, int nfeat, int grid_size, float min, float max) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= nbatch || i >= nfeat) return;
    
    int flat_idx = i * nbatch + b;
    // Load as T, convert to float for math
    float x = static_cast<float>(inputs[flat_idx]);
    float step = (max - min) / static_cast<float>(grid_size);
    
    float x_clamped = x;
    if (x_clamped < min) x_clamped = min;
    if (x_clamped > max) x_clamped = max - 1e-5f;
    
    float grid_pos = (x_clamped - min) / step;
    int k = static_cast<int>(floorf(grid_pos));
    if (k < 0) k = 0; 
    if (k > grid_size - 1) k = grid_size - 1;
    
    float u = grid_pos - static_cast<float>(k);
    float b_val[4];
    compute_cubic_basis<T>(u, b_val);
    
    int out_ptr = (i * nbatch * 4) + (b * 4);
    // Write back as T
    basis_out[out_ptr + 0] = static_cast<T>(b_val[0]); 
    basis_out[out_ptr + 1] = static_cast<T>(b_val[1]);
    basis_out[out_ptr + 2] = static_cast<T>(b_val[2]); 
    basis_out[out_ptr + 3] = static_cast<T>(b_val[3]);
    index_out[flat_idx] = k;
}

// 2. Forward Kernel
template <typename T>
__global__ void forward_kernel(
    const T* __restrict__ basis, 
    const int* __restrict__ index, 
    const T* __restrict__ weights, 
    T* __restrict__ output, 
    int nbatch, int nfeat, int nout, int ncoeffs) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= nbatch || o >= nout) return;
    
    // Accumulate in float for precision, even if T is half
    float acc = 0.0f;
    for (int i = 0; i < nfeat; i++) {
        int flat_idx = i * nbatch + b;
        int k = index[flat_idx];
        int basis_ptr = (flat_idx * 4);
        int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
        
        // Cast __ldg reads to float
        acc += static_cast<float>(basis[basis_ptr + 0]) * static_cast<float>(__ldg(&weights[w_ptr + 0]));
        acc += static_cast<float>(basis[basis_ptr + 1]) * static_cast<float>(__ldg(&weights[w_ptr + 1]));
        acc += static_cast<float>(basis[basis_ptr + 2]) * static_cast<float>(__ldg(&weights[w_ptr + 2]));
        acc += static_cast<float>(basis[basis_ptr + 3]) * static_cast<float>(__ldg(&weights[w_ptr + 3]));
    }
    output[o * nbatch + b] = static_cast<T>(acc);
}

// 3. Backward Weights (Shared Memory Tiled)
template <typename T>
__global__ void backward_weights_shared_kernel(
    const T* __restrict__ grad_out, 
    const T* __restrict__ basis, 
    const int* __restrict__ index, 
    T* __restrict__ grad_weights, 
    int nbatch, int nfeat, int nout, int ncoeffs) 
{
    extern __shared__ char smem[];
    T* s_grads = reinterpret_cast<T*>(smem);

    int in_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    int tid = threadIdx.x;

    for (int k = tid; k < ncoeffs; k += blockDim.x) {
        s_grads[k] = static_cast<T>(0.0f);
    }
    __syncthreads();

    for (int b = tid; b < nbatch; b += blockDim.x) {
        float gy = static_cast<float>(grad_out[out_idx * nbatch + b]);
        int flat_idx = in_idx * nbatch + b;
        int k = index[flat_idx];
        int basis_ptr = flat_idx * 4;

        // Math in float -> cast result to T -> fast_atomic_add
        fast_atomic_add(&s_grads[k + 0], static_cast<T>(gy * static_cast<float>(basis[basis_ptr + 0])));
        fast_atomic_add(&s_grads[k + 1], static_cast<T>(gy * static_cast<float>(basis[basis_ptr + 1])));
        fast_atomic_add(&s_grads[k + 2], static_cast<T>(gy * static_cast<float>(basis[basis_ptr + 2])));
        fast_atomic_add(&s_grads[k + 3], static_cast<T>(gy * static_cast<float>(basis[basis_ptr + 3])));
    }
    __syncthreads();

    int w_offset = (out_idx * nfeat * ncoeffs) + (in_idx * ncoeffs);
    for (int k = tid; k < ncoeffs; k += blockDim.x) {
        fast_atomic_add(&grad_weights[w_offset + k], s_grads[k]);
    }
}

// 4. Backward Inputs (Direct Atomics)
template <typename T>
__global__ void backward_inputs_kernel(
    const T* __restrict__ grad_out, 
    const T* __restrict__ inputs, 
    const T* __restrict__ weights, 
    T* __restrict__ grad_inputs, 
    T* __restrict__ grad_min, 
    T* __restrict__ grad_max, 
    int nbatch, int nfeat, int nout, int grid_size, int ncoeffs, 
    float min, float max) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= nbatch || i >= nfeat) return;
    
    int flat_idx = i * nbatch + b;
    float x = static_cast<float>(inputs[flat_idx]);
    float step = (max - min) / static_cast<float>(grid_size);

    if (x >= min && x <= max) {
        float x_clamped = x;
        if (x_clamped > max) x_clamped = max - 1e-5f;
        
        float grid_pos = (x_clamped - min) / step;
        int k = static_cast<int>(floorf(grid_pos));
        if (k < 0) k = 0; 
        if (k > grid_size - 1) k = grid_size - 1;
        
        float u = grid_pos - static_cast<float>(k);
        float db[4];
        compute_cubic_derivative<T>(u, db); 
        
        float acc = 0.0f;
        for (int o = 0; o < nout; o++) {
            float gy = static_cast<float>(grad_out[o * nbatch + b]);
            int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
            
            float dot = db[0] * static_cast<float>(__ldg(&weights[w_ptr + 0])) + 
                        db[1] * static_cast<float>(__ldg(&weights[w_ptr + 1])) + 
                        db[2] * static_cast<float>(__ldg(&weights[w_ptr + 2])) + 
                        db[3] * static_cast<float>(__ldg(&weights[w_ptr + 3]));
            acc += gy * dot;
        }
        
        float grad_x = acc * (1.0f / step);
        grad_inputs[flat_idx] = static_cast<T>(grad_x);

        float x_norm = (x - min) / (max - min);
        float d_min = grad_x * (x_norm - 1.0f);
        float d_max = grad_x * (-x_norm);

        fast_atomic_add(grad_min, static_cast<T>(d_min));
        fast_atomic_add(grad_max, static_cast<T>(d_max));
    } else {
        grad_inputs[flat_idx] = static_cast<T>(0.0f);
    }
}

// ==========================================================================
// PYBIND11 (With HALF Support)
// ==========================================================================
std::vector<torch::Tensor> compute_basis(torch::Tensor inputs, int grid_size, double min, double max) {
    int num_features = inputs.size(0); 
    int num_batch = inputs.size(1);
    auto basis = torch::empty({num_features, num_batch, 4}, inputs.options());
    auto index = torch::empty({num_features, num_batch}, inputs.options().dtype(torch::kInt32));
    dim3 threads(16, 16);
    dim3 blocks((num_batch + 15)/16, (num_features + 15)/16);
    // CHANGE: AT_DISPATCH_FLOATING_TYPES_AND_HALF
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.scalar_type(), "basis", ([&] {
        basis_kernel<scalar_t><<<blocks, threads>>>(inputs.data_ptr<scalar_t>(), basis.data_ptr<scalar_t>(), index.data_ptr<int>(), num_batch, num_features, grid_size, static_cast<float>(min), static_cast<float>(max));
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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(basis.scalar_type(), "forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(basis.data_ptr<scalar_t>(), index.data_ptr<int>(), weights.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_batch, num_features, num_outputs, num_coeffs);
    }));
    return output;
}

torch::Tensor run_backward_weights(torch::Tensor grad_out, torch::Tensor basis, torch::Tensor index, int O, int I, int C) {
    int num_batch = grad_out.size(1);
    auto grad_weights = torch::zeros({O, I, C}, grad_out.options());
    dim3 grid(I, O);
    int threads = 256;
    int smem_size = C * grad_out.element_size(); 
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "bw_weights", ([&] {
        backward_weights_shared_kernel<scalar_t><<<grid, threads, smem_size>>>(grad_out.data_ptr<scalar_t>(), basis.data_ptr<scalar_t>(), index.data_ptr<int>(), grad_weights.data_ptr<scalar_t>(), num_batch, I, O, C);
    }));
    return grad_weights;
}

std::vector<torch::Tensor> run_backward_inputs(torch::Tensor grad_out, torch::Tensor inputs, torch::Tensor weights, int grid_size, double min, double max) {
    int num_features = inputs.size(0); 
    int num_batch = inputs.size(1); 
    int num_outputs = weights.size(0); 
    int num_coeffs = weights.size(2);
    auto grad_inputs = torch::zeros_like(inputs);
    auto grad_min = torch::zeros({1}, inputs.options());
    auto grad_max = torch::zeros({1}, inputs.options());
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_features+15)/16);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.scalar_type(), "bw_inputs", ([&] {
        backward_inputs_kernel<scalar_t><<<blocks, threads>>>(grad_out.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>(), grad_min.data_ptr<scalar_t>(), grad_max.data_ptr<scalar_t>(), num_batch, num_features, num_outputs, grid_size, num_coeffs, static_cast<float>(min), static_cast<float>(max));
    }));
    return {grad_inputs, grad_min, grad_max};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_basis", &compute_basis);
    m.def("run_forward", &run_forward);
    m.def("run_backward_weights", &run_backward_weights);
    m.def("run_backward_inputs", &run_backward_inputs);
}