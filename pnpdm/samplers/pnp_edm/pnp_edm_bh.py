import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from .denoiser_edm import Denoiser_EDM

class PnPEDMBH:
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.device = device
        if config.mode == 'vp':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
        elif config.mode == 've':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'iddpm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='pfode')
        elif config.mode == 'edm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
        elif config.mode == 'vp_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
        elif config.mode == 've_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='sde')
        elif config.mode == 'iddpm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
        elif config.mode == 'edm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='sde')
        else:
            raise NotImplementedError(f"Mode {self.config.mode} is not implemented.")

    @property
    def display_name(self):
        return f'pnp-{self.config.mode}-rho={self.config.rho}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        torch.manual_seed(873345)
        log = defaultdict(list)
        cmap = 'afmhot_10us'
        x = self.operator.initialize(gt, y_n, 1)

        # logging
        x_save = inv_transform(x)
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())
        
        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1)
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1)
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1)
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1)

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        path = os.path.join('deleteme', save_root, 'progress')
        os.makedirs(path, exist_ok=True)
        plt.imsave(path + f'/init.png', x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)

        samples = []
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters, 
            self.config.num_iters-1, 
            self.config.num_samples_per_run+1, 
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter)

            # prior step
            if i % 5 == 0:
                plt.imsave(path + f'/iter{i}_z.png', z_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
                # repeat z 10 times along the batch dimension
                z_batch = z.repeat(10, 1, 1, 1)
                x_batch = self.edm(z_batch, rho_iter)
                # save all x's
                for j in range(10):
                    x_save = inv_transform(x_batch[j:j+1])
                    plt.imsave(path + f'/iter{i}_x_{j}.png', x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
            if i == 50:
                raise
            x = self.edm(z, rho_iter)

            if i in iters_count_as_sample:
                samples.append(x)

            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')
            
            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save), dim=-1)
                zs_save = torch.cat((zs_save, z_save), dim=-1)
                
            if i % 5 == 0:
                plt.imsave(path + f'/iter{i}_x.png', x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
                plt.imsave(path + f'/iter{i}_z.png', z_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr"])
        plt.title(f'psnr (max): {np.amax(log["psnr"]):.4f}, (last): {log["psnr"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim"])
        plt.title(f'ssim (max): {np.amax(log["ssim"]):.4f}, (last): {log["ssim"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips"])
        plt.title(f'lpips (min): {np.amin(log["lpips"]):.4f}, (last): {log["lpips"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        # logging
        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return torch.concat(samples, dim=0)


class PnPEDMBHBatch(PnPEDMBH):
    @property
    def display_name(self):
        return f'pnp-{self.config.mode}-bh-batch-rho={self.config.rho}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        torch.manual_seed(873345)
        cmap = 'afmhot_10us'
        log = defaultdict(list)
        x = self.operator.initialize(gt, y_n, self.config.batch_size)
        
        # logging
        x_save = inv_transform(x.mean(0, keepdim=True))
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())
        
        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1)
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1)
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1)
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1)

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        samples = []
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters, 
            self.config.num_iters-1, 
            self.config.num_samples_per_run+1, 
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"

        # logging
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter)

            # prior step
            x = self.edm(z, rho_iter)

            if i in iters_count_as_sample:
                samples.append(x)

            # logging
            x_save = inv_transform(x.mean(0, keepdim=True))
            z_save = inv_transform(z.mean(0, keepdim=True))
            x_std_save = inv_transform(x.std(0, keepdim=True))
            z_std_save = inv_transform(z.std(0, keepdim=True))
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')
            
            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save), dim=-1)
                zs_save = torch.cat((zs_save, z_save), dim=-1)
                
            if i % 10 == 0:
                path = os.path.join('deleteme', save_root, 'progress')
                os.makedirs(path, exist_ok=True)
                plt.imsave(path + f'/iter{i}_x_std.png', x_std_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
                plt.imsave(path + f'/iter{i}_z_std.png', z_std_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
                plt.imsave(path + f'/iter{i}_x.png', x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
                plt.imsave(path + f'/iter{i}_z.png', z_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr"])
        plt.title(f'psnr (max): {np.amax(log["psnr"]):.4f}, (last): {log["psnr"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim"])
        plt.title(f'ssim (max): {np.amax(log["ssim"]):.4f}, (last): {log["ssim"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips"])
        plt.title(f'lpips (min): {np.amin(log["lpips"]):.4f}, (last): {log["lpips"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        # logging
        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        log['samples'] = torch.concat(samples, dim=0).permute(0, 2, 3, 1).cpu().numpy()
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return x