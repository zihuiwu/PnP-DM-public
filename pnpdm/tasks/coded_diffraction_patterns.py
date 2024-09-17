import torch
import numpy as np
from torch.fft import fftn, ifftn
from . import register_operator, NonLinearOperator

@register_operator(name='coded_diffraction_patterns')
class CodedDiffractionPatterns(NonLinearOperator):
    def __init__(self, num_measurements, channels, img_dim, device) -> None:
        self.num_measurements = num_measurements
        self.shape = channels, img_dim, img_dim
        np.random.seed(23)
        Areal = np.random.rand(num_measurements, *self.shape)
        Aimag = np.sqrt(1 - Areal ** 2)
        Areal = Areal * (-1) ** np.random.randint(10, size=Areal.shape)
        Aimag = Aimag * (-1) ** np.random.randint(10, size=Aimag.shape)
        self.Amat = torch.from_numpy(Areal + 1j*Aimag).to(device)

    @property
    def display_name(self):
        return 'cdp'

    def forward(self, x, **kwargs):
        x = (x + 1.0) / 2.0 # [-1, 1] -> [0, 1]
        return self._Amult(x).abs()
    
    def likelihood_gradient(self, x, y, sigma):
        z = self._Amult(x)
        res = z - y * z / z.abs()
        grad = self._Atran(res).sum(dim=0)
        assert not(torch.any(torch.isnan(grad))), 'grad contains nan'
        return grad.real.type(torch.float32) / sigma**2

    def _Amult(self, x):
        fft_dims = tuple(1+np.arange(len(self.shape)))
        z = fftn(self.Amat * x, dim=fft_dims, norm='ortho')
        return z
    
    def _Atran(self, z):
        ifft_dims = tuple(1+np.arange(len(self.shape)))
        x = self.Amat.conj() * ifftn(z, dim=ifft_dims, norm='ortho')
        return x

    def proximal_generator(self, x, y, sigma, rho, gamma=1e-3, num_iters=100):
        z = x
        z.requires_grad_()
        for _ in range(num_iters):
            data_fit = (self.forward(z) - y).norm()**2 / (2*sigma**2)
            grad = torch.autograd.grad(outputs=data_fit, inputs=z)[0]
            z = z - gamma * grad - (gamma/rho**2) * (z - x) + np.sqrt(2*gamma) * torch.randn_like(x)
        return z.type(torch.float32)

    def proximal_for_admm(self, x, y, rho, gamma=1e-3, num_iters=100):
        x = (x + 1.0) / 2.0 # [-1, 1] -> [0, 1]
        rho /= 2.0 # to compensate for the scaling of x
        z = x
        for _ in range(num_iters):
            z = z - gamma * self.likelihood_gradient(z, y, 1.0) - (gamma*rho) * (z - x)
        return z * 2.0 - 1.0 # [0, 1] -> [-1, 1]

    def initialize(self, gt, y):
        return torch.randn_like(gt)