import numpy as np
import torch
from discrete import diffusion_utils
from discrete import network_preprocess


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, device, noise='cos'):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps
        # 得到 beta 列
        # if noise_schedule == 'cosine':
        #     betas = diffusion_utils.custom_beta_schedule_discreteDig(timesteps)
        # elif noise_schedule == 'liner':
        #     betas = diffusion_utils.custom_beta_schedule_discreteDig(timesteps)
        # else:
        #     raise NotImplementedError(noise_schedule)
        # betas = betas * 0.7
        if noise == 'cos':
            betas = diffusion_utils.custom_beta_schedule_discreteDig(timesteps)
        else:
            spec = {'num_timesteps': self.timesteps, 'start': 0, 'stop': 1, 'type': 'linear'}
            betas = diffusion_utils.get_diffusion_betas(spec)
        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.betas = self.betas.to(device)

        # alpha = 1 - beta, 阶段alpha在[0,0.9999]
        # self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        self.alphas = 1 - self.betas
        # log(alpha)
        # log_alpha = torch.log(self.alphas)
        # log_alpha_bar = torch.cumsum(log_alpha, dim=0)  # cumsum(log(alpha))
        # self.alphas_bar = torch.exp(log_alpha_bar)      # e^cumsum(log(alpha))
        # log_alpha = torch.log(self.alphas)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # cumsum(alpha)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]


class DiscreteUniformTransition:
    def __init__(self, e_classes: int):
        self.E_classes = e_classes
        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_e = self.u_e.to(device)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)

        return network_preprocess.PlaceHolder(X=None, E=q_e, y=None)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e

        return network_preprocess.PlaceHolder(X=None, E=q_e, y=None)


class MarginalUniformTransition:
    def __init__(self, e_marginals):
        self.E_classes = len(e_marginals)
        self.e_marginals = e_marginals
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)

        return network_preprocess.PlaceHolder(X=None, E=q_e, y=None)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e

        return network_preprocess.PlaceHolder(X=None, E=q_e, y=None)