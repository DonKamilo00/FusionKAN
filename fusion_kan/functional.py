import torch
import torch.cuda.amp as amp # Import AMP

# 1. Try to import the compiled C++ extension
try:
    import _fusion_kan_cuda as _backend
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    pass

class FusionKANFunction(torch.autograd.Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float16) # Cast inputs to FP16 automatically in AMP context
    def forward(ctx, inputs, weights, grid_size, grid_min_val, grid_max_val):
        if not KERNEL_AVAILABLE:
            raise RuntimeError("FusionKAN CUDA kernel not found. Ensure 'pip install .' was successful.")
        
        # Ensure contiguous
        inputs_T = inputs.transpose(0, 1).contiguous()
        
        # Grid bounds are scalars, keep them as is (usually FP32 in config)
        g_min = grid_min_val.item()
        g_max = grid_max_val.item()
        
        basis_T, index_T = _backend.compute_basis(inputs_T, grid_size, g_min, g_max)
        output_T = _backend.run_forward(basis_T, index_T, weights)
        
        ctx.save_for_backward(basis_T, index_T, weights, inputs_T)
        ctx.grid_size = grid_size
        ctx.grid_min = g_min
        ctx.grid_max = g_max
        
        return output_T.transpose(0, 1)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_out):
        basis_T, index_T, weights, inputs_T = ctx.saved_tensors
        grad_out_T = grad_out.transpose(0, 1).contiguous()
        
        # 1. Weights Gradient
        grad_weights = _backend.run_backward_weights(
            grad_out_T, basis_T, index_T, 
            weights.size(0), weights.size(1), weights.size(2)
        )
        
        # 2. Inputs Gradient
        grad_inputs = None
        grad_min = None
        grad_max = None
        
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            grads = _backend.run_backward_inputs(
                grad_out_T, inputs_T, weights, 
                ctx.grid_size, ctx.grid_min, ctx.grid_max
            )
            
            grad_inputs = grads[0].transpose(0, 1)
            grad_min = grads[1]
            grad_max = grads[2]
            
        return grad_inputs, grad_weights, None, grad_min, grad_max