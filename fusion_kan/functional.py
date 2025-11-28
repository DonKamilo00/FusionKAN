import torch

# 1. Try to import the compiled C++ extension
try:
    import _fusion_kan_cuda as _backend
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    pass

class FusionKANFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, grid_size, grid_min_val, grid_max_val):
        # grid_min_val and grid_max_val are Tensors (scalars) so autograd tracks them.
        
        if not KERNEL_AVAILABLE:
            raise RuntimeError("FusionKAN CUDA kernel not found. Ensure 'pip install .' was successful.")
        
        # Ensure inputs are [Features, Batch] and contiguous
        inputs_T = inputs.transpose(0, 1).contiguous()
        
        # Pass values to CUDA kernel (basis calc depends on these)
        g_min = grid_min_val.item()
        g_max = grid_max_val.item()
        
        basis_T, index_T = _backend.compute_basis(inputs_T, grid_size, g_min, g_max)
        output_T = _backend.run_forward(basis_T, index_T, weights)
        
        # Save context
        ctx.save_for_backward(basis_T, index_T, weights, inputs_T)
        ctx.grid_size = grid_size
        ctx.grid_min = g_min
        ctx.grid_max = g_max
        
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
        
        # 2. Inputs, Min, Max Gradients
        grad_inputs = None
        grad_min = None
        grad_max = None
        
        # We calculate input gradients if any input-related tensor needs them
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            # backend returns [grad_inputs_T, grad_min, grad_max]
            grads = _backend.run_backward_inputs(
                grad_out_T, inputs_T, weights, 
                ctx.grid_size, ctx.grid_min, ctx.grid_max
            )
            
            grad_inputs_T = grads[0]
            grad_min = grads[1]
            grad_max = grads[2]
            
            # Transpose inputs back to [Batch, Features]
            grad_inputs = grad_inputs_T.transpose(0, 1)
            
        # Return matching forward signature: 
        # inputs, weights, grid_size, grid_min, grid_max
        return grad_inputs, grad_weights, None, grad_min, grad_max