import torch
from torch import Tensor
import torch.nn as nn
import math


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

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

def expmap0_poincare(u, curvature):
    """
    Exponential map: map points from the tangent space at the vertex
    to the hyperboloid using the exponential map of poincare ball model.
    """
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(curvature ** 0.5 * u_norm) * u / (curvature ** 0.5 * u_norm)
    return gamma_1

def logmap0_poincare(y, curvature):
    """
    Logarithmic map: map points from the hyperboloid to the tangent space
    at the vertex using the logarithmic map of poincare ball model.
    """
    sqrt_c = curvature ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

def expmap0_lorentz(x: Tensor, curvature: float = 0.01, eps: float = 1e-8) -> Tensor:
    """
    Exponential map: map points from the tangent space at the vertex
    to the hyperboloid using the exponential map of Lorentz model.
    """
    #if torch.norm(x) < eps:
    #    return torch.zeros_like(x)
    rc_xnorm = curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
    _output = torch.sinh(sinh_input) * x / rc_xnorm
    return _output

def logmap0_lorentz(x: Tensor, curvature: float = 0.01, eps: float = 1e-8) -> Tensor:
    """
    Logarithmic map: map points from the hyperboloid to the tangent space
    at the vertex using the logarithmic map of Lorentz model.
    """
    #if torch.norm(x) < eps:
    #    return torch.zeros_like(x)
    rc_x_time = torch.sqrt(1 + curvature * torch.sum(x ** 2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def lorentz_distance(x: Tensor, y: Tensor, curvature: float = 0.01, eps: float = 1e-8) -> Tensor:
    """
    Compute the Lorentz distance between points x and y on the hyperboloid.
    
    In the Lorentz model, points are represented in the hyperboloid:
    -x_0^2 + x_1^2 + ... + x_n^2 = -1/c
    
    The distance formula is:
    d(x, y) = (1/sqrt(c)) * acosh(-c * <x, y>_L)
    
    where <x, y>_L is the Lorentz inner product: -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
    
    Args:
        x: First point tensor of shape (..., n)
        y: Second point tensor of shape (..., n)
        curvature: Curvature parameter (positive for hyperbolic space)
        eps: Small constant for numerical stability
    
    Returns:
        Distance tensor of shape (..., 1)
    """
    # Compute the time components (x_0, y_0) from the spatial components
    # From the hyperboloid equation: x_0 = sqrt(1/c + c * ||x||^2)
    sqrt_c = curvature ** 0.5
    
    x_sqnorm = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_sqnorm = torch.sum(y ** 2, dim=-1, keepdim=True)
    
    x_time = torch.sqrt(1 / curvature + x_sqnorm)
    y_time = torch.sqrt(1 / curvature + y_sqnorm)
    
    # Compute Lorentz inner product: -x_0*y_0 + <x, y>
    spatial_inner_product = torch.sum(x * y, dim=-1, keepdim=True)
    lorentz_inner_product = -x_time * y_time + spatial_inner_product
    
    # Distance formula: d = (1/sqrt(c)) * acosh(-c * <x, y>_L)
    # Note: -c * <x, y>_L should be >= 1 for valid points on the hyperboloid
    acosh_input = -curvature * lorentz_inner_product
    acosh_input = torch.clamp(acosh_input, min=1.0 + eps)
    
    dist = (1 / sqrt_c) * torch.acosh(acosh_input)
    
    return dist

def poincare_distance(x, y, curvature):
    sq_dist = torch.sum((x - y).pow(2), dim=-1, keepdim=True)
    x_sqnorm = torch.sum(x.pow(2), dim=-1, keepdim=True)
    y_sqnorm = torch.sum(y.pow(2), dim=-1, keepdim=True)

    denom_x = 1 - curvature * x_sqnorm
    denom_y = 1 - curvature * y_sqnorm

    denom_x = torch.clamp(denom_x, min=1e-7)
    denom_y = torch.clamp(denom_y, min=1e-7)

    gamma = 1 + 2 * curvature * sq_dist / (denom_x * denom_y)
    gamma = torch.clamp(gamma, min=1.0 + 1e-7)

    sqrt_c = curvature ** 0.5
    dist = (1 / sqrt_c) * torch.acosh(gamma)

    return dist

# def mobius_add(x, y, curvature):
#     x2 = x.pow(2).sum(dim=-1, keepdim=True)
#     y2 = y.pow(2).sum(dim=-1, keepdim=True)
#     xy = (x * y).sum(dim=-1, keepdim=True)
#     num = (1 + 2 * curvature * xy + curvature * y2) * x + (1 - curvature * x2) * y
#     denom = 1 + 2 * curvature * xy + curvature ** 2 * x2 * y2
#     return num / (denom + 1e-5)

# def poincare_distance(x, y, curvature):
#     sqrt_c = curvature ** 0.5
#     mobius_sum = mobius_add(-x, y, curvature)
#     mobius_norm = torch.norm(mobius_sum, dim=-1, p=2)
#     mobius_norm = torch.clamp(mobius_norm, min=1e-10, max=(1.0 - 1e-5) / sqrt_c)
#     return (2 / sqrt_c) * artanh(sqrt_c * mobius_norm)


class AngleCalculator(nn.Module):
    def __init__(self, curvature: float = 0.1):
        super().__init__()
        self.curvature = curvature

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()

    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def forward(self, text_embeddings, image_embeddings):
        """
        Compute average angle between text embeddings and image embeddings.
        
        Args:
            text_embeddings: shape [N_text, 1, D] e.g., [171, 1, 512]
            image_embeddings: shape [B, D, H, W] e.g., [4, 512, 24, 24]
        Returns:
            angle: scalar tensor - average angle across all text-image pairs
        """
        # text: [171, 1, 512] -> [171, 512]
        text_emb = text_embeddings.squeeze(1)
        
        # image: [B, D, H, W] -> [B*H*W, D]
        B, D, H, W = image_embeddings.shape
        img_emb = image_embeddings.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        
        # Map to hyperbolic space
        text_hyperbolic = self.expmap0(text_emb)  # [171, 512]
        image_hyperbolic = self.expmap0(img_emb)  # [B*H*W, 512]
        
        # Compute average pooled image embedding per batch
        img_hyp_pooled = image_hyperbolic.reshape(B, H*W, D).mean(dim=1)  # [B, 512]
        
        # Compute angle between each text and averaged image embedding
        # text: [171, 512], img: [B, 512]
        # For simplicity, average across batch dimension too
        img_hyp_avg = img_hyp_pooled.mean(dim=0)  # [512]
        
        # Compute angles for all text embeddings
        dot = torch.sum(text_hyperbolic * img_hyp_avg.unsqueeze(0), dim=-1)  # [171]
        text_norm = torch.norm(text_hyperbolic, dim=-1)  # [171]
        image_norm = torch.norm(img_hyp_avg)  # scalar
        
        cos_angle = dot / (text_norm * image_norm + 1e-8)  # [171]
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
        
        angles = torch.acos(cos_angle)  # [171]
        
        # Return mean angle as scalar
        return angles.mean()