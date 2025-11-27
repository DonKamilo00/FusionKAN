import torch

# 1. Try to import the CUDA kernel
try:
    import fusion_kan
    KERNEL_AVAILABLE = True
except ImportError:
    # This happens on your laptop!
    KERNEL_AVAILABLE = False
    # Don't crash here, just warn or pass
    pass

class FusionKANFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, grid_size, grid_min, grid_max):
        # 2. Check availability at Runtime
        if not KERNEL_AVAILABLE:
            raise RuntimeError("FusionKAN CUDA kernel not found. Are you on a GPU machine?")
        # Ensure inputs are [Features, Batch] and contiguous
        # The C++ kernel expects transposed input for coalesced access
        inputs_T = inputs.transpose(0, 1).contiguous()
        
        basis_T, index_T = fusion_kan.compute_basis(inputs_T, grid_size, grid_min, grid_max)
        output_T = fusion_kan.run_forward(basis_T, index_T, weights)
        
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
        grad_weights = fusion_kan.run_backward_weights(
            grad_out_T, basis_T, index_T, 
            weights.size(0), weights.size(1), weights.size(2)
        )
        
        # 2. Inputs Gradient
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs_T = fusion_kan.run_backward_inputs(
                grad_out_T, inputs_T, weights, 
                ctx.grid_size, ctx.grid_min, ctx.grid_max
            )
            # Transpose back to [Batch, Features]
            grad_inputs = grad_inputs_T.transpose(0, 1)
            
        return grad_inputs, grad_weights, None, None, None