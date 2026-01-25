import torch
from torch import Tensor
import torch.nn as nn


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

def expmap0(u, curvature):
    """
    Exponential map: map points from the tangent space at the vertex
    to the hyperboloid using the exponential map of poincare ball model.
    """
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(curvature ** 0.5 * u_norm) * u / (curvature ** 0.5 * u_norm)
    return gamma_1

def logmap0(y, curvature):
    """
    Logarithmic map: map points from the hyperboloid to the tangent space
    at the vertex using the logarithmic map of poincare ball model.
    """
    sqrt_c = curvature ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

def mobius_add(x, y, curvature):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * curvature * xy + curvature * y2) * x + (1 - curvature * x2) * y
    denom = 1 + 2 * curvature * xy + curvature ** 2 * x2 * y2
    return num / (denom + 1e-5)

def poincare_distance(x, y, curvature):
    sqrt_c = curvature ** 0.5
    mobius_sum = mobius_add(-x, y, curvature)
    mobius_norm = torch.norm(mobius_sum, dim=-1, p=2)
    mobius_norm = torch.clamp(mobius_norm, min=1e-10, max=(1.0 - 1e-5) / sqrt_c)
    return (2 / sqrt_c) * artanh(sqrt_c * mobius_norm)


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