import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft2, ifft2, fftshift
from . import register_operator, LinearOperator
from .modules.kernel import Kernel


@register_operator(name='motion_blur_circ')
class MotionBlurCircular(LinearOperator):
    def __init__(self, kernel_size, intensity, channels, img_dim, device, seed=None) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.img_dim = img_dim
        self.device = device
        self.generate_kernel_(seed)

    def generate_kernel_(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.intensity)
        self.kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        pre1 = (self.img_dim-self.kernel.shape[0])//2
        post1 = self.img_dim-self.kernel.shape[0]-pre1
        pre2 = (self.img_dim-self.kernel.shape[1])//2
        post2 = self.img_dim-self.kernel.shape[1]-pre2
        self.full_kernel = torch.nn.functional.pad(self.kernel, (pre1, post1, pre2, post2), "constant", 0).type(torch.complex64).to(self.device)
        self.full_spectrum = fft2(self.full_kernel)[None, None]

    @property
    def display_name(self):
        return 'mblur-circ'

    def forward(self, x, **kwargs):
        return fftshift(ifft2(self.full_spectrum * fft2(x)).real, dim=(-2,-1))

    def transpose(self, y):
        return fftshift(ifft2(torch.conj(self.full_spectrum) * fft2(y)).real, dim=(-2,-1))
     
    def proximal_generator(self, x, y, sigma, rho):
        power = self.full_spectrum * torch.conj(self.full_spectrum)
        inv_spectrum = 1 / ((power / sigma**2) + (torch.ones_like(x) / rho**2))
        noise = ifft2(torch.sqrt(inv_spectrum) * fft2(torch.randn_like(x))).real
        mu_x = ifft2(inv_spectrum * fft2(self.transpose(y / sigma**2) + (x / rho**2))).real
        return mu_x + noise

    def proximal_for_admm(self, x, y, rho):
        power = self.full_spectrum * torch.conj(self.full_spectrum)
        inv_spectrum = 1 / (power + rho)
        return ifft2(inv_spectrum * fft2(self.transpose(y) + rho * x)).real

    def initialize(self, gt, y):
        return torch.zeros_like(gt)