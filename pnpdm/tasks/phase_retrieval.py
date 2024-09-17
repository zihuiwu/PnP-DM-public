import torch
import numpy as np
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift
from . import register_operator, NonLinearOperator


@register_operator(name='phase_retrieval')
class PhaseRetrieval(NonLinearOperator):
    def __init__(self, oversample, device):
        self.oversample = oversample
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
    
    @property
    def display_name(self):
        return f'pr-os={self.oversample}'
        
    def forward(self, x, **kwargs):
        x = (x + 1.0) / 2.0 # [-1, 1] -> [0, 1]
        padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def proximal_generator(self, x, y, sigma, rho, gamma=1e-4, num_iters=100):
        z = x
        z.requires_grad_()
        for _ in range(num_iters):
            data_fit = (self.forward(z) - y).norm()**2 / (2*sigma**2)
            grad = torch.autograd.grad(outputs=data_fit, inputs=z)[0]
            z = z - gamma * grad - (gamma/rho**2) * (z - x) + np.sqrt(2*gamma) * torch.randn_like(x)
        return z.type(torch.float32)

    def proximal_for_admm(self, x, y, rho, gamma=1e-4, num_iters=100):
        z = x
        z.requires_grad_()
        for _ in range(num_iters):
            data_fit = (self.forward(z) - y).norm()**2
            grad = torch.autograd.grad(outputs=data_fit, inputs=z)[0]
            z = z - gamma * grad - (gamma*rho) * (z - x)
        return z.type(torch.float32)

    def initialize(self, gt, y):
        return torch.randn_like(gt)


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex128)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex128)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    
    data = fftshift(data, dim=[-3, -2])

    return data