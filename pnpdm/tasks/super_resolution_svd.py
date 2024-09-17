import torch
from . import register_operator, LinearSVDOperator


@register_operator(name='super_resolution_svd')
class SuperResolution(LinearSVDOperator):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    @property
    def display_name(self):
        return f'sr{self.ratio:d}-svd'

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        singulars = self.singulars_small
        
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        
        lambda_t = torch.ones(self.ratio ** 2, device=vec.device)
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)
            
        lambda_t = lambda_t.reshape(1, 1, 1, -1)
#         print("lambda_t:", lambda_t)
#         print("V:", self.V_small)
#         print(lambda_t.size(), self.V_small.size())
#         print("Sigma_t:", torch.matmul(torch.matmul(self.V_small, torch.diag(lambda_t.reshape(-1))), self.Vt_small))
        patches = patches * lambda_t
        
        
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))
        
        patches = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small
        
        patches_vec = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_vec = patches_vec.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches_eps = epsilon.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_eps = patches_eps.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        d1_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index  * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)
             
            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)
            
            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5
        
        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t
        
        patches_vec = torch.matmul(self.V_small, patches_vec.reshape(-1, self.ratio**2, 1))
        
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        patches_eps = torch.matmul(self.V_small, patches_eps.reshape(-1, self.ratio**2, 1))
        
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches_vec + patches_eps

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
