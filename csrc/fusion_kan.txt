#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ==========================================================================
// MATH HELPERS
// ==========================================================================
template <typename T>
__device__ __forceinline__ void compute_cubic_basis(T u, T* b) {
    T u2 = u * u; T u3 = u2 * u;
    b[0] = (1.0 - u3 + 3.0*u2 - 3.0*u) / 6.0;
    b[1] = (3.0*u3 - 6.0*u2 + 4.0) / 6.0;
    b[2] = (-3.0*u3 + 3.0*u2 + 3.0*u + 1.0) / 6.0;
    b[3] = u3 / 6.0;
}

template <typename T>
__device__ __forceinline__ void compute_cubic_derivative(T u, T* db) {
    T u2 = u * u;
    db[0] = (-3.0*u2 + 6.0*u - 3.0) / 6.0;
    db[1] = (9.0*u2 - 12.0*u) / 6.0;
    db[2] = (-9.0*u2 + 6.0*u + 3.0) / 6.0;
    db[3] = 3.0*u2 / 6.0;
}

// ==========================================================================
// KERNELS (Forward, Backward Weights, Backward Inputs)
// ==========================================================================

// 1. Basis Kernel
template <typename T>
__global__ void basis_kernel(const T* __restrict__ inputs, T* __restrict__ basis_out, int* __restrict__ index_out, int nbatch, int nfeat, int grid_size, T min, T max) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= nbatch || i >= nfeat) return;
    
    int flat_idx = i * nbatch + b;
    T x = inputs[flat_idx];
    T step = (max - min) / static_cast<T>(grid_size);
    
    T x_clamped = x;
    if (x_clamped < min) x_clamped = min;
    if (x_clamped > max) x_clamped = max - static_cast<T>(1e-5);
    
    T grid_pos = (x_clamped - min) / step;
    int k = static_cast<int>(floor(grid_pos));
    if (k < 0) k = 0; if (k > grid_size - 1) k = grid_size - 1;
    
    T u = grid_pos - static_cast<T>(k);
    T b_val[4];
    compute_cubic_basis(u, b_val);
    
    int out_ptr = (i * nbatch * 4) + (b * 4);
    basis_out[out_ptr + 0] = b_val[0]; basis_out[out_ptr + 1] = b_val[1];
    basis_out[out_ptr + 2] = b_val[2]; basis_out[out_ptr + 3] = b_val[3];
    index_out[flat_idx] = k;
}

// 2. Forward Kernel
template <typename T>
__global__ void forward_kernel(const T* __restrict__ basis, const int* __restrict__ index, const T* __restrict__ weights, T* __restrict__ output, int nbatch, int nfeat, int nout, int ncoeffs) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= nbatch || o >= nout) return;
    
    T acc = 0.0;
    for (int i = 0; i < nfeat; i++) {
        int k = index[i * nbatch + b];
        int basis_ptr = (i * nbatch * 4) + (b * 4);
        int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
        acc += basis[basis_ptr + 0] * weights[w_ptr + 0] + basis[basis_ptr + 1] * weights[w_ptr + 1] +
               basis[basis_ptr + 2] * weights[w_ptr + 2] + basis[basis_ptr + 3] * weights[w_ptr + 3];
    }
    output[o * nbatch + b] = acc;
}

// 3. Backward Weights
template <typename T>
__global__ void backward_weights_kernel(const T* __restrict__ grad_out, const T* __restrict__ basis, const int* __restrict__ index, T* __restrict__ grad_weights, int nbatch, int nfeat, int nout, int ncoeffs) {
    extern __shared__ float s_accum[];
    int i = blockIdx.y;
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    
    for (int t = tid; t < nout * ncoeffs; t += blockDim.x) s_accum[t] = 0.0f;
    __syncthreads();
    
    int stride = blockDim.x * gridDim.x;
    int start = (chunk * blockDim.x) + tid;
    
    for (int b = start; b < nbatch; b += stride) {
        int basis_ptr = (i * nbatch * 4) + (b * 4);
        T b0 = basis[basis_ptr + 0]; T b1 = basis[basis_ptr + 1]; T b2 = basis[basis_ptr + 2]; T b3 = basis[basis_ptr + 3];
        int k = index[i * nbatch + b];
        for (int o = 0; o < nout; o++) {
            T gy = grad_out[o * nbatch + b];
            int s_ptr = (o * ncoeffs) + k;
            atomicAdd(&s_accum[s_ptr + 0], static_cast<float>(gy * b0));
            atomicAdd(&s_accum[s_ptr + 1], static_cast<float>(gy * b1));
            atomicAdd(&s_accum[s_ptr + 2], static_cast<float>(gy * b2));
            atomicAdd(&s_accum[s_ptr + 3], static_cast<float>(gy * b3));
        }
    }
    __syncthreads();
    
    for (int t = tid; t < nout * ncoeffs; t += blockDim.x) {
        int o = t / ncoeffs; int c = t % ncoeffs;
        if (abs(s_accum[t]) > 1e-9) {
            int g_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + c;
            atomicAdd(&grad_weights[g_ptr], static_cast<T>(s_accum[t]));
        }
    }
}

// 4. Backward Inputs
template <typename T>
__global__ void backward_inputs_kernel(const T* __restrict__ grad_out, const T* __restrict__ inputs, const T* __restrict__ weights, T* __restrict__ grad_inputs, int nbatch, int nfeat, int nout, int grid_size, int ncoeffs, T min, T max) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= nbatch || i >= nfeat) return;
    
    T step = (max - min) / static_cast<T>(grid_size);
    int flat_idx = i * nbatch + b;
    T x = inputs[flat_idx];
    
    if (x < min || x > max) { grad_inputs[flat_idx] = 0.0; return; }
    
    T x_clamped = x;
    if (x_clamped > max) x_clamped = max - static_cast<T>(1e-5);
    T grid_pos = (x_clamped - min) / step;
    int k = static_cast<int>(floor(grid_pos));
    if (k < 0) k = 0; if (k > grid_size - 1) k = grid_size - 1;
    
    T u = grid_pos - static_cast<T>(k);
    T db[4];
    compute_cubic_derivative(u, db);
    
    T acc = 0.0;
    for (int o = 0; o < nout; o++) {
        T gy = grad_out[o * nbatch + b];
        int w_ptr = (o * nfeat * ncoeffs) + (i * ncoeffs) + k;
        T dot = db[0] * weights[w_ptr + 0] + db[1] * weights[w_ptr + 1] + db[2] * weights[w_ptr + 2] + db[3] * weights[w_ptr + 3];
        acc += gy * dot;
    }
    grad_inputs[flat_idx] = acc * (static_cast<T>(1.0) / step);
}

// ==========================================================================
// PYTHON BINDINGS
// ==========================================================================

std::vector<torch::Tensor> compute_basis(torch::Tensor inputs, int grid_size, double min, double max) {
    int num_features = inputs.size(0); int num_batch = inputs.size(1);
    auto basis = torch::empty({num_features, num_batch, 4}, inputs.options());
    auto index = torch::empty({num_features, num_batch}, inputs.options().dtype(torch::kInt32));
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch + 15)/16, (num_features + 15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "basis", ([&] {
        basis_kernel<scalar_t><<<blocks, threads>>>(
            inputs.data_ptr<scalar_t>(), basis.data_ptr<scalar_t>(), index.data_ptr<int>(),
            num_batch, num_features, grid_size, static_cast<scalar_t>(min), static_cast<scalar_t>(max));
    }));
    return {basis, index};
}

torch::Tensor run_forward(torch::Tensor basis, torch::Tensor index, torch::Tensor weights) {
    int num_features = basis.size(0); int num_batch = basis.size(1); int num_outputs = weights.size(0); int num_coeffs = weights.size(2);
    auto output = torch::zeros({num_outputs, num_batch}, basis.options());
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_outputs+15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            basis.data_ptr<scalar_t>(), index.data_ptr<int>(), weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), num_batch, num_features, num_outputs, num_coeffs);
    }));
    return output;
}

torch::Tensor run_backward_weights(torch::Tensor grad_out, torch::Tensor basis, torch::Tensor index, int O, int I, int C) {
    int num_batch = grad_out.size(1);
    auto grad_weights = torch::zeros({O, I, C}, grad_out.options());
    
    int chunks = 16;
    dim3 grid(chunks, I);
    int threads = 256;
    int shared_mem = O * C * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "bw_weights", ([&] {
        backward_weights_kernel<scalar_t><<<grid, threads, shared_mem>>>(
            grad_out.data_ptr<scalar_t>(), basis.data_ptr<scalar_t>(), index.data_ptr<int>(),
            grad_weights.data_ptr<scalar_t>(), num_batch, I, O, C);
    }));
    return grad_weights;
}

torch::Tensor run_backward_inputs(torch::Tensor grad_out, torch::Tensor inputs, torch::Tensor weights, int grid_size, double min, double max) {
    int num_features = inputs.size(0); int num_batch = inputs.size(1); int num_outputs = weights.size(0); int num_coeffs = weights.size(2);
    auto grad_inputs = torch::zeros_like(inputs);
    
    dim3 threads(16, 16);
    dim3 blocks((num_batch+15)/16, (num_features+15)/16);
    
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "bw_inputs", ([&] {
        backward_inputs_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
            grad_inputs.data_ptr<scalar_t>(), num_batch, num_features, num_outputs, grid_size, num_coeffs, static_cast<scalar_t>(min), static_cast<scalar_t>(max));
    }));
    return grad_inputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_basis", &compute_basis, "Compute Spline Basis");
    m.def("run_forward", &run_forward, "Run Forward Pass");
    m.def("run_backward_weights", &run_backward_weights, "Run Backward Weights");
    m.def("run_backward_inputs", &run_backward_inputs, "Run Backward Inputs");
}