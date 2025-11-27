import torch

# 1. Try to import the compiled C++ extension
# Note: In setup.py, the extension name is '_fusion_kan_cuda'
try:
    import _fusion_kan_cuda as _backend
    KERNEL_AVAILABLE = True
except ImportError:
    # This happens if the code is run on a machine without the compiled extension
    KERNEL_AVAILABLE = False
    pass

class FusionKANFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, grid_size, grid_min, grid_max):
        # 2. Check availability at Runtime
        if not KERNEL_AVAILABLE:
            raise RuntimeError("FusionKAN CUDA kernel not found. Ensure 'pip install .' was successful and you are on a GPU machine.")
        
        # Ensure inputs are [Features, Batch] and contiguous
        inputs_T = inputs.transpose(0, 1).contiguous()
        
        # CALL THE BACKEND (Not fusion_kan)
        basis_T, index_T = _backend.compute_basis(inputs_T, grid_size, grid_min, grid_max)
        output_T = _backend.run_forward(basis_T, index_T, weights)
        
        ctx.save_for_backward(basis_T, index_T, weights, inputs_T)
        ctx.grid_size = grid_size
        ctx.grid_min = grid_min
        ctx.grid_max = grid_max
        
        # Output is [Outputs, Batch], transpose back to [Batch, Outputs]
        return output_T.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_out):
        basis_T, index_T, weights, inputs_T = ctx.saved_tensors
        
        # Transpose gradient to match [Outputs, Batch] layout
        grad_out_T = grad_out.transpose(0, 1).contiguous()
        
        # 1. Weights Gradient
        grad_weights = _backend.run_backward_weights(
            grad_out_T, basis_T, index_T, 
            weights.size(0), weights.size(1), weights.size(2)
        )
        
        # 2. Inputs Gradient
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs_T = _backend.run_backward_inputs(
                grad_out_T, inputs_T, weights, 
                ctx.grid_size, ctx.grid_min, ctx.grid_max
            )
            # Transpose back to [Batch, Features]
            grad_inputs = grad_inputs_T.transpose(0, 1)
            
        return grad_inputs, grad_weights, None, None, None