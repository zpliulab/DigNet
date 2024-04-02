import torch
from torch.nn import functional as F
import numpy as np
import math
from discrete.network_preprocess import PlaceHolder
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, roc_curve
from torchmetrics import Metric, MeanSquaredError
from torch import Tensor
import pandas as pd
import os

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return np.linspace(spec['start'], spec['stop'], spec['num_timesteps'])
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
            np.arange(spec['num_timesteps'] + 1, dtype=np.float64) /
            spec['num_timesteps'])
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / np.linspace(spec['num_timesteps'], 1., spec['num_timesteps'])
    else:
        raise NotImplementedError(spec.type)


def custom_beta_schedule_discreteDig(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps+1
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    # assert timesteps >= 100
    #
    # p = 4 / 5       # 1 - 1 / num_edge_classes
    # num_edges = average_num_nodes * (average_num_nodes - 1) / 2
    #
    # # First 100 steps: only a few updates per graph
    # updates_per_graph = 1.2
    # beta_first = updates_per_graph / (p * num_edges)
    #
    # betas[betas < beta_first] = beta_first
    return np.array(betas)


def sample_discrete_max(X, probE, node_mask, test=False):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = X.shape
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    E_t = torch.argmax(probE, dim=-1)
    E_t = delte_dig_from_batch(E_t)
    return PlaceHolder(X=X, E=E_t, y=None)


def sample_discrete_features(X, probE, node_mask, randomseed=42, test=True):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = X.shape
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = torch.multinomial(probE, num_samples=1, replacement=True).reshape(bs, n, n)  # (bs, n, n)
    E_t = delte_dig_from_batch(E_t)

    return PlaceHolder(X=X, E=E_t, y=None)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def Evaluation(y_true, y_pred, flag=False):
    if isinstance(y_pred, torch.Tensor):
        y_p = y_pred.cpu().detach().numpy()
    else:
        y_p = np.array(y_pred)
 #   y_p = y_p.flatten()
    y_t = y_true.cpu().numpy().flatten().astype(int)
    AUC = roc_auc_score(y_true=y_t, y_score=y_p, average=None)

    fpr, tpr, thresholds = roc_curve(y_true=y_t, y_score=y_p)
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    precision, recall, thresholds = precision_recall_curve(y_true=y_t, probas_pred=y_p)
    AUPR = auc(recall, precision)
    AUPR_norm = AUPR/np.mean(y_t)

    y_p[y_p > best_threshold] = 1
    y_p[y_p != 1] = 0
    f1 = f1_score(y_t, y_p)
    return {'AUC': AUC, 'AUPR': AUPR, 'AUPR_norm': AUPR_norm, 'F1': f1}


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d 这行其实是针对边特征来说的，输入边的特征为 512*9*9*5，转换后为512*81*4
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def sample_discrete_feature_noise(X, limit_dist, node_mask, seed=42):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    if seed is not None:
        torch.manual_seed(seed)
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
        #torch.seed()
    else:
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
    U_E = delte_dig_from_batch(U_E)
    long_mask = node_mask.long()
    U_E = U_E.type_as(long_mask)
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
    y = torch.empty(1, 0).type_as(U_E)
    return PlaceHolder(X=X, E=U_E, y=y).mask(node_mask)


def delte_dig_from_batch(batchdata):
    for i in range(0, batchdata.shape[0]):
        a = batchdata[i,:,:]
        a = a.fill_diagonal_(1)
        batchdata[i, :, :] = a
    return batchdata


def cal_del_TF_edge(GeneName):
    TF_list = pd.read_csv('GRN/TF.txt', sep='\t')
    GeneName = pd.Series(GeneName)
    TF_list = TF_list[TF_list['Symbol'].isin(GeneName)]['Symbol']
    TF_positions = GeneName[GeneName.isin(TF_list)]
    original_list = list(range(GeneName.shape[0]))
    GENE_ID_list = [x for x in original_list if x not in TF_positions.index]
    TF_ID_list = list(TF_positions.index)
    return GENE_ID_list, TF_ID_list