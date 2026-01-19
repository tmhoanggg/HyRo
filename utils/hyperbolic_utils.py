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


class AngleCalculator(nn.Module):
    def __init__(self, curvature: int = 0.1):
        self.curvature = curvature

    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        #sqrt_c = self.curvature ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        Logarithmic map for :math:`y` from :math:`0` on the manifold.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def forward(self, text_embeddings, image_embeddings):
        text_hyperbolic = self.expmap0(text_embeddings)
        image_hyperbolic = self.expmap0(image_embeddings)

        # Calculate angle at the origin
        dot = torch.sum(text_hyperbolic * image_hyperbolic, dim=-1)
        text_norm = torch.norm(text_hyperbolic, dim=-1)
        image_norm = torch.norm(image_hyperbolic, dim=-1)

        cos_angle = dot / (text_norm * image_norm + 1e-8)
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)

        angle = torch.acos(cos_angle)
        return angle