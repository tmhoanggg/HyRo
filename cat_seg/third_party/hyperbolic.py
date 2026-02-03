import clip.model
import torch.nn as nn
import clip
from torch.nn import functional as F
import torch
from einops import rearrange
from . import model_hyperbolic
import math
from torch import Tensor


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


class BlockDiagonalLinear_text(nn.Module):
    def __init__(self, block_size, in_features, out_features, curvature: float = 0.01, rot_block_size: int = 256):
        super(BlockDiagonalLinear_text, self).__init__()
        self.block_size = block_size
        self.r = int(out_features / block_size)
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        # 创建对角分块矩阵的权重
        self.weights = nn.Parameter(torch.stack([
            torch.diag(torch.ones(block_size)) for _ in range(self.r)
        ]))

        # Angle Adjustment Parameter
        self.rot_block_size = rot_block_size
        self.r_rot = int(out_features / self.rot_block_size)
        self.rotation_weights = nn.Parameter(
            torch.zeros(self.r_rot, rot_block_size, rot_block_size)
        )


    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A
    
    def get_orthogonal_matrix(self):
        """
        Generates an orthogonal matrix R from the unconstrained parameter theta.
        R = exp(A), where A = theta - theta^T (Skew-Symmetric)
        """          
        A = self.rotation_weights - self.rotation_weights.transpose(-1, -2)
        
        # Option 1: Basic formulation
        # R = torch.matrix_exp(A)
        
        # Option 2: Use Cayley transform
        # I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        # R = torch.linalg.solve(I - A, I + A)

        # Option 3: Use block-diagonal with Cayley transform
        I = torch.eye(self.rot_block_size, device=A.device, dtype=A.dtype).unsqueeze(0)
        R_blocks = torch.linalg.solve(I - A, I + A)
        R = torch.block_diag(*R_blocks)
        return R

    def exp_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
        _output = torch.sinh(sinh_input) * x / rc_xnorm
        return _output
    
    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        #sqrt_c = self.curvature ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def log_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_x_time = torch.sqrt(1 + self.curvature * torch.sum(x ** 2, dim=-1, keepdim=True))
        _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
        return _output

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        Logarithmic map for :math:`y` from :math:`0` on the manifold.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def mobius_matvec(self, m, x):
        r"""
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        .. math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Parameters
        ----------
        m : tensor
            matrix for multiplication
        x : tensor
            point on poincare ball
        c : float|tensor
            negative ball curvature
        Returns
        -------
        tensor
            Mobius matvec result
        """
        #c = torch.as_tensor(c).type_as(x)
        return self._mobius_matvec(m, x)


    def _mobius_matvec(self, m, x):
        x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        sqrt_c = self.curvature ** 0.5
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
        res_c = self.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_0, res_c)
        return self._project(res)
    
    def _project(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        maxnorm = (1 - 1e-3) / (self.curvature ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    def mobius_add(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.curvature * xy + self.curvature * y2) * x + (1 - self.curvature * x2) * y
        denom = 1 + 2 * self.curvature * xy + self.curvature ** 2 * x2 * y2
        return num / (denom + 1e-5)

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
    
    def save_hyperbolic_embeddings(self, emb_before, emb_after):
        """
        Saves the intermediate embeddings to a .pt file, appending to existing data.
        """
        import os
        import torch.distributed as dist
        
        # 1. Get rank to ensure safe writing in distributed settings
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return
        
        os.makedirs("insight", exist_ok=True)
        save_path = f'insight/text_embeddings_rank{rank}.pt'

        # 2. Detach and move to CPU (Critical for saving)
        # We assume the user wants the full batch appended as a tensor in a list
        val_before = emb_before.detach().cpu()
        val_after = emb_after.detach().cpu()

        # 3. Load existing data or start fresh
        if os.path.exists(save_path):
            try:
                # weights_only=False is used in your example
                data = torch.load(save_path, weights_only=False)
            except (EOFError, RuntimeError, Exception):
                # If file is corrupted, start over
                data = {}
        else:
            data = {}

        # 4. Initialize lists if keys don't exist
        if 'before_rotation' not in data:
            data['before_rotation'] = []
        if 'after_rotation' not in data:
            data['after_rotation'] = []

        # 5. Append new batch
        data['before_rotation'].append(val_before)
        data['after_rotation'].append(val_after)

        # 6. Save back to disk
        torch.save(data, save_path)


    def forward(self, x, visual=False):
        # Ensure input shape is (batch_size, 512, 512)
        output_hyperbolic = self.expmap0(x)
        fix_filt = output_hyperbolic.data
        orig_dtype = fix_filt.dtype
        block_diagonal_weight = self.block_diagonal(self.weights)
        output_hyperbolic_filt = self.mobius_matvec(block_diagonal_weight.to(orig_dtype), fix_filt)
        rotation_weights = self.get_orthogonal_matrix().to(orig_dtype) # Add: Rotation
        output_hyperbolic_rotated = output_hyperbolic_filt @ rotation_weights.transpose(-1, -2) # Add: rotate
        
        #self.save_hyperbolic_embeddings(output_hyperbolic_filt, output_hyperbolic_rotated)

        output_euclidean = self.logmap0(output_hyperbolic_rotated)
        return output_euclidean




class BlockDiagonalLinear(nn.Module):
    def __init__(self, block_size, in_features, out_features, curvature: float = 2.5, rot_block_size: int = 256):
        super(BlockDiagonalLinear, self).__init__()
        self.block_size = block_size
        self.r = int(out_features / block_size)
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature #nn.Parameter(torch.tensor(curvature), requires_grad=True)
        # 创建对角分块矩阵的权重
        self.weights = nn.Parameter(torch.stack([
            torch.diag(torch.ones(block_size)) for _ in range(self.r)
        ]))

        # Angle Adjustment Parameter
        self.rot_block_size = rot_block_size
        self.r_rot = out_features // self.rot_block_size
        self.rotation_weights = nn.Parameter(
            torch.zeros(self.r_rot, rot_block_size, rot_block_size)
        )

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A
    
    def get_orthogonal_matrix(self):
        """
        Generates an orthogonal matrix R from the unconstrained parameter theta.
        R = exp(A), where A = theta - theta^T (Skew-Symmetric)
        """
        A = self.rotation_weights - self.rotation_weights.transpose(-1, -2)
        #R = torch.matrix_exp(A)
        # Use Cayley transform
        # I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        # R = torch.linalg.solve(I - A, I + A)

        I = torch.eye(self.rot_block_size, device=A.device, dtype=A.dtype).unsqueeze(0)
        R_blocks = torch.linalg.solve(I - A, I + A)
        R = torch.block_diag(*R_blocks)
        return R

    def exp_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of Lorentz model.
        """
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
        _output = torch.sinh(sinh_input) * x / rc_xnorm
        return _output

    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def log_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of Lorentz model.
        """
        rc_x_time = torch.sqrt(1 + self.curvature * torch.sum(x ** 2, dim=-1, keepdim=True))
        _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
        return _output

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        Logarithmic map for :math:`y` from :math:`0` on the manifold.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
       

    def mobius_matvec(self, m, x):
        r"""
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        .. math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Parameters
        ----------
        m : tensor
            matrix for multiplication
        x : tensor
            point on poincare ball
        c : float|tensor
            negative ball curvature
        Returns
        -------
        tensor
            Mobius matvec result
        """
        #c = torch.as_tensor(c).type_as(x)
        return self._mobius_matvec(m, x)


    def _mobius_matvec(self, m, x):
        x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        sqrt_c = self.curvature ** 0.5
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
        res_c = self.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_0, res_c)
        return self._project(res)

    def mobius_add(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.curvature * xy + self.curvature * y2) * x + (1 - self.curvature * x2) * y
        denom = 1 + 2 * self.curvature * xy + self.curvature ** 2 * x2 * y2
        return num / (denom + 1e-5)
    
    def _project(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        maxnorm = (1 - 1e-3) / (self.curvature ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
    
    def save_hyperbolic_embeddings(self, emb_before, emb_after):
        """
        Saves the intermediate embeddings to a .pt file, appending to existing data.
        """
        import os
        import torch.distributed as dist
        
        # 1. Get rank to ensure safe writing in distributed settings
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Only save from last layer (count == 11)
        if not hasattr(self, 'count') or self.count != 11:
            return

        os.makedirs("insight", exist_ok=True)
        save_path = f'insight/visual_embeddings_rank{rank}.pt'

        # 2. Detach and move to CPU (Critical for saving)
        # We assume the user wants the full batch appended as a tensor in a list
        val_before = emb_before.detach().cpu()
        val_after = emb_after.detach().cpu()

        # 3. Load existing data or start fresh
        if os.path.exists(save_path):
            try:
                data = torch.load(save_path, weights_only=False)
            except (EOFError, RuntimeError, Exception):
                # If file is corrupted, start over
                data = {}
        else:
            data = {}

        # 4. Initialize lists if keys don't exist
        if 'before_rotation' not in data:
            data['before_rotation'] = []
        if 'after_rotation' not in data:
            data['after_rotation'] = []

        # 5. Append new batch
        data['before_rotation'].append(val_before)
        data['after_rotation'].append(val_after)

        # 6. Save back to disk
        torch.save(data, save_path)


    def forward(self, x, visual=False):
        output_hyperbolic = self.expmap0(x)
        fix_filt = output_hyperbolic.data
        orig_dtype = fix_filt.dtype
        block_diagonal_weight = self.block_diagonal(self.weights)
        output_hyperbolic_filt_stretch = self.mobius_matvec(block_diagonal_weight.to(orig_dtype), fix_filt)
        rotation_weights = self.get_orthogonal_matrix().to(orig_dtype) # Add: Rotation weights
        output_hyperbolic_rotated = output_hyperbolic_filt_stretch @ rotation_weights.transpose(-1, -2) # Add: rotate
        
        #self.save_hyperbolic_embeddings(output_hyperbolic_filt_stretch, output_hyperbolic_rotated)
        
        output_euclidean = self.logmap0(output_hyperbolic_rotated)
        return output_euclidean


def forward_attn_init(self, x):
    B, N, C = x.shape
    res_x = x
    orig_dtype = x.dtype
    in_proj_weight_new = self.hyperbolic_attn(self.attn.in_proj_weight)
    qkv = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 1, 3, 0, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
    attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(1).cuda().to(orig_dtype)
    attn = attn.softmax(dim=-1)

    # ADD THIS: Store attention weights for visualization
    if hasattr(self, 'save_attention') and self.save_attention:
        self.attention_map = attn.detach()

    final_lora = ((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C)
    final_lora = self.attn.out_proj(final_lora)
    final_lora = self.dp(final_lora)
    final = res_x + final_lora #* self.s
    final = final + self.mlp(self.ln_2(final))
    return final


def oft_forward_vision_init(self, x):
    B, N, C = x.shape
    res_x = x
    if self.count <= self.last_layer_index:
        in_proj_weight_new = self.hyperbolic_attn(self.attn.in_proj_weight, visual=True)
        qkv = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
        attn = attn.softmax(dim=-1)
        final_lora = ((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C)
        final_lora = self.attn.out_proj(final_lora)
        final_lora = self.dp(final_lora)
        final = res_x + final_lora #* self.s
        final = final + self.mlp(self.ln_2(final))
        return final
    
    else:
        in_proj_weight_new = self.hyperbolic_attn(self.attn.in_proj_weight, visual=True)
        y = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias)
        L, N, D = y.shape # L N 3D        
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)
        y = self.attn.out_proj(y)        
        q, k, v = y.tensor_split(3, dim=0)      
        v = v.transpose(1, 0) + x[:1] # L N D
        v = v + self.mlp(self.ln_2(v))
        return v



class Adapter_init(nn.Module):
    def __init__(self, hidden_size, dim, dim_rot, curvature_ratio=1.0):
        super().__init__()

        self.adapter_attn_q = BlockDiagonalLinear(block_size=dim, rot_block_size=dim_rot,
                                                  in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.adapter_attn_k = BlockDiagonalLinear(block_size=dim, rot_block_size=dim_rot,
                                                  in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.adapter_attn_v = BlockDiagonalLinear(block_size=dim, rot_block_size=dim_rot,
                                                  in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.dim = dim


    def forward(self, attn, visual=False):
        #B, N, C = attn.shape
        orig_dtype = attn.dtype

        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)

        filt_q = self.adapter_attn_q(q_proj_weight, visual=visual)

        filt_k = self.adapter_attn_k(k_proj_weight, visual=visual) 

        filt_v = self.adapter_attn_v(v_proj_weight, visual=visual)

        filt  = torch.cat([filt_q, filt_k, filt_v], dim=0)
        return filt.to(orig_dtype)


class Adapter_init_text(nn.Module):
    def __init__(self, hidden_size, dim, dim_rot, curvature_ratio=1.0):
        super().__init__()

        self.adapter_attn_q = BlockDiagonalLinear_text(block_size=dim, rot_block_size=dim_rot,
                                                       in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.adapter_attn_k = BlockDiagonalLinear_text(block_size=dim, rot_block_size=dim_rot,
                                                       in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.adapter_attn_v = BlockDiagonalLinear_text(block_size=dim, rot_block_size=dim_rot,
                                                       in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.dim = dim


    def forward(self, attn, visual=False):
        #B, N, C = attn.shape
        orig_dtype = attn.dtype

        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)

        filt_q = self.adapter_attn_q(q_proj_weight, visual=visual)

        filt_k = self.adapter_attn_k(k_proj_weight, visual=visual) 

        filt_v = self.adapter_attn_v(v_proj_weight, visual=visual)

        filt  = torch.cat([filt_q, filt_k, filt_v], dim=0)
        return filt.to(orig_dtype)




def set_adapter_hyperbolic(model, dim=32, dim_rot=32, hidden_size=512, s=0.1, count=0):
    print(dim)
    for _ in model.children():
        if type(_) == model_hyperbolic.ResidualAttentionBlock:
            print('count', count, 's', s)
            print(_)
            _.hyperbolic_attn = Adapter_init_text(hidden_size, dim, dim_rot)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            count+=1
            _.count = count
            bound_method = forward_attn_init.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_adapter_hyperbolic(_, dim, dim_rot, hidden_size, s, count)
    print('count',count)

def set_adapter_vision_hyperbolic(model, dim=32, dim_rot=32, hidden_size=768, s=0.1, count=0, curvature_ratio=0.01):
    for _ in model.children():
        if type(_) == model_hyperbolic.ResidualAttentionBlock:
            real_hidden_size = _.attn.embed_dim
            last_layer_index = 23 if real_hidden_size == 1024 else 11 # ViT-B/16 -> last index 11; ViT-L/14 -> last index 23
            _.last_layer_index = last_layer_index

            _.hyperbolic_attn = Adapter_init(hidden_size, dim, dim_rot, curvature_ratio)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            count+=1
            _.count = count
            bound_method = oft_forward_vision_init.__get__(_, _.__class__)

            print('count', count, 's', s)
            print(_)
            if count < last_layer_index:
                setattr(_, 'forward', bound_method)
            else:
                setattr(_, 'forward_dense', bound_method)
        elif len(list(_.children())) != 0:
            set_adapter_vision_hyperbolic(_, dim, dim_rot, hidden_size, s, count, curvature_ratio)
    print('count',count)