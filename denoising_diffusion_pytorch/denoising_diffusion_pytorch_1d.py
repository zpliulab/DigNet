from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from discrete.noise_predefined import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, MarginalUniformTransition
from discrete import network_preprocess
from discrete import diffusion_utils
from discrete.models.train_metrics import TrainLossDiscrete
from discrete.diffusion_utils import SumExceptBatchKL, NLL, SumExceptBatchMetric, cal_del_TF_edge
from torch_geometric.data import Data, Batch
import torch_geometric.utils as utils
from discrete.diffusion_utils import Evaluation
import copy
import pandas as pd

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RowNormalize(nn.Module):
    def forward(self, x):
        # 计算每行的范数
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # 归一化每行
        normalized_x = x / norm
        # 确保每行的两个数加起来为1
        normalized_x = normalized_x / torch.sum(normalized_x, dim=1, keepdim=True)
        return normalized_x


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        edge_percent=0.3,
        max_num_nodes=None,
        device='cpu',
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        predflag = 'E_start',
        transition='marginal',  # 转移矩阵类型
        ddim_sampling_eta=0.,
        noise_type='cos',
    ):
        super().__init__()
        noise_schedule = 'cosine'  # 噪音曲线

        self.Edim_output = 2       # 边的类型数量
        self.device = device
        self.train_loss = TrainLossDiscrete()
        self.val_E_kl = SumExceptBatchKL()
        self.val_nll = NLL()
        self.val_E_logp = SumExceptBatchMetric()
        self.model = model
        self.predflag = predflag
        self.edge_percent = edge_percent
        if max_num_nodes is not None:
            self.max_num_nodes = max_num_nodes
        self.normalize_layer = RowNormalize()
        # 预定义噪音表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule,
                                                              timesteps=timesteps,
                                                              device=self.device,
                                                              noise=noise_type)
        # 预定义转移矩阵Q
        if transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(e_classes=self.Edim_output)
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            self.limit_dist = network_preprocess.PlaceHolder(X=None, E=e_limit, y=None)

        elif transition == 'marginal':
            e_marginals = torch.tensor([1- self.edge_percent,  self.edge_percent], dtype=torch.float32, device=self.device )
            self.transition_model = MarginalUniformTransition(e_marginals=e_marginals)
            self.limit_dist = network_preprocess.PlaceHolder(X=None, E=e_marginals, y=None)

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

    def noisy_data2data_dense(self, noisy_data, all_edge=True):
        data_list = []
        data_list_ALL = []
        for bs_i in range(0, noisy_data['X_t'].shape[0]):
            edge_weights = noisy_data['E_t'][bs_i, :, :, :]
            node_features = noisy_data['X_t'][bs_i, :, :]
            # 生成全连接图的边索引
            num_nodes = node_features.size(0)
            edge_index = utils.dense_to_sparse(torch.ones(num_nodes, num_nodes))[0].to(self.device)
            data_ALL = Data(x=node_features, edge_attr=edge_weights.reshape(-1, edge_weights.shape[2]))
            # 设置边权重属性
            data_ALL.edge_index = edge_index
            # sparse
            edge_weights0 = edge_weights[:, :, 0]
            # 从邻接矩阵中获取边的索引
            indices_tensor = edge_weights0.nonzero(as_tuple=False).t().contiguous()
            num_edges = indices_tensor.shape[1]
            values_tensor = torch.tensor([[1, 0]], dtype=torch.float32).repeat(num_edges, 1)
            data = Data(x=node_features, edge_index=indices_tensor, edge_attr=values_tensor)
            data_list_ALL.append(data_ALL)
            data_list.append(data)

        batch_ALL = Batch.from_data_list(data_list_ALL)
        batch_ALL.edge_index.to(self.device)
        ALLedge_index = batch_ALL.edge_index
        batch = Batch.from_data_list(data_list)
        batch.edge_index.to(self.device)
        if all_edge:
            return batch_ALL, ALLedge_index
        else:
            return batch, ALLedge_index


    def pred_shape2data(self, pred, noisy_data):
        bs, n, _, c = noisy_data['E_t'].shape
        if isinstance(pred, torch.Tensor):
            pred = pred.reshape(bs, n, n, c)
            return network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=pred, y=noisy_data['y_t'])
        else:
            pred.E = pred.E.reshape(bs, n, n, c)
            return network_preprocess.PlaceHolder(X=pred.X, E=pred.E, y=pred.y)


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')


    def p_losses(self, E, noisy_data, node_mask, loss_method='celoss', calAUC=True):
        """
            E: 一个干净的原始数据
            noisy_data: 加噪数据
            node_mask: 掩码矩阵
        """
        batch, ALLedge_index = self.noisy_data2data_dense(noisy_data, all_edge=False)
        time_tensor_tonode = noisy_data['t'][batch.batch].squeeze(-1)
    #    time_tensor_toedge = noisy_data['t_int'].repeat_interleave(self.max_num_nodes * self.max_num_nodes)
        torch.set_grad_enabled(True)
        pred = self.model(batch.x, batch.edge_attr, time_tensor_tonode, batch.edge_index, ALLedge_index, node_mask, noisy_data)
        pred = self.pred_shape2data(pred, noisy_data)
        mask_pred_probs_E = pred.mask(node_mask)
        mask_pred_probs_ES = F.softmax(mask_pred_probs_E.E, dim=-1)
        # mask_pred_probs_ES = F.sigmoid(mask_pred_probs_E.E)

        if self.predflag =='E_start':
            target = E
        else:
            target = noisy_data['E_s']
        if loss_method == 'celoss':
            loss = self.train_loss(masked_pred_E=mask_pred_probs_ES, nosoft_pred_E=mask_pred_probs_E.E, true_E=target)
        else:
            loss = self.loss_xtpxt(E, pred.E, noisy_data, node_mask)

        if calAUC:
            # 预测结果：计算Xt-1
            E_pred_onehot = self.given_pred_to_edge(mask_pred_probs_E, noisy_data, node_mask, predflah2='weight')  # 根据预测的x0，计算Xt-1
            E_pred_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=E_pred_onehot, y=noisy_data['y_t'])         # (bs,n,n,2)
            E_pred_discrete = E_pred_onehot.mask(node_mask, collapse=True)

            # 真实结果： 计算Xt-1
            if self.predflag =='E_start':
                _, _, E_true_discrete_view = self.vb_terms_bpd(E, pred.E, noisy_data, node_mask)
                E_true_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=E_true_discrete_view, y=noisy_data['y_t'])  # (bs,n,n,2)
            else:
                E_true_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_s'],
                                                               y=noisy_data['y_t'])  # (bs,n,n,2)
            E_true_discrete = E_true_onehot.mask(node_mask, collapse=True)
            AUC_list = []
            for bsi in range(E.shape[0]):
                true1 = E_true_discrete.E[bsi, :, :]
                true1 = true1[:, node_mask[bsi, :]]
                true1 = true1[node_mask[bsi, :], :]
                true1 = true1.flatten()

                pred1 = E_pred_discrete.E[bsi, :, :]
                pred1 = pred1[:, node_mask[bsi, :]]
                pred1 = pred1[node_mask[bsi, :], :]
                pred1 = pred1.flatten()
                performance = Evaluation(y_pred=pred1, y_true=true1)
                AUC_list.append(performance['AUC'])
            AUC = np.min(AUC_list)
            return loss, pred, AUC
        else:
            return loss, pred


    def apply_noise(self, X, E, y, node_mask,t_int=None):
        """ Sample noise and apply it to the data. """
        if t_int is None:
            t_int = torch.randint(0, self.num_timesteps, size=(X.shape[0], 1), device=self.device).float()
        # 2. 得到s = t-1
        s_int = t_int - 1
        mask = (s_int == -1)
        s_int[mask] = 0

        # 3.
        t_float = t_int / self.num_timesteps
        s_float = s_int / self.num_timesteps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_int=t_int)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s_int)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=t_int)      # (bs, 1)

        # 计算 t时刻的加噪数据
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()                         # 确保概率转移矩阵的总和为1
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)        # Compute transition probabilities
        sampled_t = diffusion_utils.sample_discrete_features(X=X, probE=probE, node_mask=node_mask)   # 采样结果，只显示最终采样结果
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (E.shape == E_t.shape)
        z_t = network_preprocess.PlaceHolder(X=X, E=E_t, y=y).type_as(X).mask(node_mask)  # z_t就是预处理后带噪音的数据

        # 计算 s时刻的加噪数据
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qsb.E.sum(dim=2) - 1.) < 1e-4).all()                         # 确保概率转移矩阵的总和为1
        probE_s = E @ Qsb.E.unsqueeze(1)  # (bs, n, n, de_out)        # Compute transition probabilities
        sampled_s = diffusion_utils.sample_discrete_features(X=X, probE=probE_s, node_mask=node_mask)   # 采样结果，只显示最终采样结果
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output)
        assert (E.shape == E_s.shape)
        z_s = network_preprocess.PlaceHolder(X=X, E=E_s, y=y).type_as(X).mask(node_mask)  # z_s就是预处理后带噪音的数据
        z_s_E = z_s.E
    #    z_s_E[mask.squeeze(), :, :, :] = E[mask.squeeze(), :, :, :]
        noisy_data = {'t_int': t_int, 't': t_float, 's_int': s_int, 's': s_float,'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'E_s': z_s_E, 'y_t': t_int, 'node_mask': node_mask}
        noisy_data['y_t'] = noisy_data['t'].float()
        return noisy_data

    def sum_except_batch(self, x):
        return x.reshape(x.size(0), -1).sum(dim=-1)


    def mask_distributions(self, true_E, pred_E, node_mask):
        # Add a small value everywhere to avoid nans
        pred_E = pred_E + 1e-7
        pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

        # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
        row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
        row_E[1] = 1.

        diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
        true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
        pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

        return true_E, pred_E


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.num_timesteps * ones-1
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        bs, n, _, _ = probE.shape

        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_E, probE = self.mask_distributions(true_E=limit_E.clone(), pred_E=probE,  node_mask=node_mask)
        probE = probE+1e-5
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return self.sum_except_batch(kl_distance_E)


    def compute_posterior_distribution(self, M, M_t, Qt_M, Qsb_M, Qtb_M):
        ''' M: X or E
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
        '''
        # Flatten feature tensors
        M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # (bs, N, d) with N = n * n
        M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

        Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

        left_term = M_t @ Qt_M_T  # (bs, N, d)
        right_term = M @ Qsb_M  # (bs, N, d)
        product = left_term * right_term  # (bs, N, d)

        denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
        denom = (denom * M_t).sum(dim=-1) +1e-7 # (bs, N, d) * (bs, N, d) + sum = (bs, N)
        prob = product / denom.unsqueeze(-1)  # (bs, N, d)

        return prob


    def posterior_distributions(self, E, E_t, Qt, Qsb, Qtb):
        prob_E = self.compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)  # (bs, n * n, de)
        return network_preprocess.PlaceHolder(X=1, E=prob_E, y=None)

    def compute_Lt(self, E, pred, noisy_data, node_mask, test=False):
        bs, n, _, _ = E.shape
        pred_probs_E = F.softmax(pred, dim=-1)
        pred_probs_E = pred_probs_E.reshape(bs, n, n, -1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        prob_true = self.posterior_distributions(E=E,
                                                 E_t=noisy_data['E_t'],
                                                 Qt=Qt,
                                                 Qsb=Qsb,
                                                 Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = self.posterior_distributions(E=pred_probs_E,
                                                 E_t=noisy_data['E_t'],
                                                 Qt=Qt,
                                                 Qsb=Qsb,
                                                 Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))
        # Reshape and filter masked rows
        prob_true_E, prob_pred.E = self.mask_distributions(true_E=prob_true.E, pred_E=prob_pred.E, node_mask=node_mask)
        kl_e = F.kl_div(torch.log(prob_pred.E+1e-7), prob_true_E, reduction='sum')
        return kl_e, prob_true_E


    def vb_terms_bpd(self, E, pred, noisy_data, node_mask):
        bs, n, _, _ = E.shape
        pred_probs_E = F.sigmoid(pred)
        pred_probs_E = pred_probs_E.reshape(bs, n, n, -1)
        pred_probs_E = self.pred_shape2data(pred_probs_E, noisy_data)
        pred_probs_E = pred_probs_E.mask(node_mask).E
        pred_probs_E0 = pred_probs_E
        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        prob_true = self.posterior_distributions(E=E,
                                                 E_t=noisy_data['E_t'],
                                                 Qt=Qt,
                                                 Qsb=Qsb,
                                                 Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))

        prob_pred_t_1 = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=noisy_data['E_t'],
                                                                                     Qt=Qt.E,
                                                                                     Qsb=Qsb.E,
                                                                                     Qtb=Qtb.E)

        pred_probs_E = pred_probs_E.reshape((bs, -1, pred_probs_E.shape[-1]))
        weighted_E = pred_probs_E.unsqueeze(-1) * prob_pred_t_1  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E_final = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])

        t0mask = (noisy_data['t_int'] == 0).squeeze(1)
        # 如果t==0，那么计算p(x0)~x0的CE，否则计算p(xt-1|xt，x0)
        if torch.any(t0mask):
            prob_E_final[t0mask, ...] = pred_probs_E0[t0mask, ...]
            prob_true.E[t0mask, ...] = E[t0mask, ...]

        prob_true_E, prob_E_final = self.mask_distributions(true_E=prob_true.E, pred_E=prob_E_final, node_mask=node_mask)
        loss = self.train_loss(masked_pred_E=prob_E_final, nosoft_pred_E=prob_E_final, true_E=prob_true_E)

        return loss, prob_E_final, prob_true_E


    def loss_xtpxt(self, E, pred, noisy_data, node_mask):
        vb_losses, pred_x_start_logits,_ = self.vb_terms_bpd(E, pred, noisy_data, node_mask)   # 缺少一个t=0的判断
        ce_losses = self.train_loss(masked_pred_E=pred_x_start_logits, nosoft_pred_E=pred_x_start_logits, true_E=E)
        losses = vb_losses + 0.001 * ce_losses
        return losses


    def reconstruction_logp(self, X, t, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)
        sampled0 = diffusion_utils.sample_discrete_features(X=X, probE=probE0, node_mask=node_mask)

        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()

        sampled_0 = network_preprocess.PlaceHolder(X=X, E=E0, y=None).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': X, 'E_t': sampled_0.E, 'y_t': None, 'node_mask': node_mask,
                      't_int': torch.zeros(X.shape[0], 1).type_as(E)}
        batch, ALLedge_index = self.noisy_data2data_dense(noisy_data, all_edge=False)
        time_tensor_tonode = noisy_data['t_int'][batch.batch].squeeze(-1)
        #    time_tensor_toedge = noisy_data['t_int'].repeat_interleave(self.max_num_nodes * self.max_num_nodes)
        pred = self.model(batch.x, batch.edge_attr, time_tensor_tonode, batch.edge_index, ALLedge_index, node_mask)
        # Normalize predictions
        probE0 = F.softmax(pred, dim=-1)
        probE0 = probE0.reshape(sampled_0.E.shape)
        # Set masked rows to arbitrary values that don't contribute to loss
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

      #  diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
      #  diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
      #  probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return network_preprocess.PlaceHolder(X=X, E=probE0, y=None)


    def compute_val_loss(self, pred, noisy_data, X, E, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = torch.log(N/self.max_num_nodes+1e-30)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t, _ = self.compute_Lt(E, pred, noisy_data, node_mask, test=False)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(X, t, E, node_mask)

        loss_term_0 = torch.sum(E * prob0.E.log())/prob0.E.shape[0]

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        # Update NLL metric object and return batch nll
        nll = torch.sum(nlls)/prob0.E.shape[0]       # Average over the batch
        return nll

    def forward(self, data, *args, **kwargs):
        """
            1. 稀疏图转为完全图（每个batch的node数量一致,不存在的edge设为[1,0,0,0,0]），生成相应的node_mask
            2. 根据node_mask,再次确认dense_data中的X和E该隐藏的点特征或者边邻接是否设置为0
            3. 最终获得完全图上的 节点特征矩阵X、邻接矩阵E、node_mask  （每个batch的这仨个维度都一样）
        """
        dense_data, node_mask, max_num_nodes = network_preprocess.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=self.max_num_nodes)
        if self.max_num_nodes is None:
            self.max_num_nodes = max_num_nodes
        dense_data = dense_data.mask(node_mask)  # 再次确认有效节点/有效边
        X, E = dense_data.X, dense_data.E
        """
            根据X、E以及Q，创建计算被污染的数据z_t(X和E)       
        """
    #    if data.y is None:
        data.y = torch.empty(data.num_graphs, 0)
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        loss, pred, AUC = self.p_losses(E, noisy_data, node_mask)
        return loss, AUC

    @torch.no_grad()
    def validation_step(self, data):
        self.model.eval()
        dense_data, node_mask, max_num_nodes = network_preprocess.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=self.max_num_nodes)
        if self.max_num_nodes is None:
            self.max_num_nodes = max_num_nodes
        dense_data = dense_data.mask(node_mask)  # 再次确认有效节点/有效边
        X, E = dense_data.X, dense_data.E
        if data.y is None:
            data.y = torch.empty(data.num_graphs, 0)
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        loss, pred = self.p_losses(E, data.edge_index, noisy_data, node_mask)
        val_loss = self.compute_val_loss(pred, noisy_data, X, E, node_mask, test=False)
        return val_loss


    def given_pred_to_edge(self, pred, noisy_data, node_mask, predflah2 = 'noweight'):
        X_t = noisy_data['X_t']
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_int=noisy_data['t_int'])  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=noisy_data['s_int'])
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=noisy_data['t_int'])
        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Normalize predictions
        pred_probs_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        pred_probs_E_sample = diffusion_utils.sample_discrete_features(X_t, pred_probs_E, node_mask=node_mask,
                                                                       test=True)
        pred_probs_E_sample.E = F.one_hot(pred_probs_E_sample.E, num_classes=self.Edim_output).float()
        pred_probs_E_sample = pred_probs_E_sample.mask(node_mask)
        pred_probs_E = pred_probs_E_sample.E

        # 去掉对角线
        diag_mask = torch.eye(n)  # 创建一个单位矩阵
        diag_mask = ~diag_mask.type_as(pred_probs_E).bool()  # 单位阵的类型=E，并且对角False，其余True
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)  # 掩码矩阵的维度与E一致
        pred_probs_E = pred_probs_E * diag_mask

        if self.predflag == 'E_start':
            if predflah2 == 'weight':
                #  p(s)<-p(t,t0)p(t0)
                prob_pred_t_1 = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=noisy_data['E_t'],
                                                                                             Qt=Qt.E,
                                                                                             Qsb=Qsb.E,
                                                                                             Qtb=Qtb.E)
                # 对edge进行加权
                pred_probs_E_re = pred_probs_E.reshape((bs, -1, pred_probs_E.shape[-1]))  # 512*81*5
                weighted_E = pred_probs_E_re.unsqueeze(-1) * prob_pred_t_1  # bs, N, d0, d_t-1
                # 归一化处理
                unnormalized_prob_E = weighted_E.sum(dim=-2)
                unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
                prob_E_final = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
                prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])
                assert ((prob_E_final.sum(dim=-1) - 1).abs() < 1e-4).all()
            else:
                #  p(s)<-p(t,t0)
                pred_probs_E = pred_probs_E.reshape(bs, n, n, -1)
                prob_true = self.posterior_distributions(E=pred_probs_E,
                                                         E_t=noisy_data['E_t'],
                                                         Qt=Qt,
                                                         Qsb=Qsb,
                                                         Qtb=Qtb)
                unnormalized_prob_E = torch.sum(prob_true.E, dim=-1, keepdim=True)
                unnormalized_prob_E[unnormalized_prob_E == 0] = 1e-5
                prob_E_final = prob_true.E / unnormalized_prob_E
                prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])
            #    assert ((prob_E_final.sum(dim=-1) - 1).abs() < 1e-3).all()
        else:
            prob_E_final = pred_probs_E / torch.sum(pred_probs_E, dim=-1, keepdim=True)

        # 如果存在t=0，则不预测xt-1，而是直接输出pred
        t0mask = (noisy_data['t_int'] == 0).squeeze(1)
        if torch.any(t0mask):
            prob_E_final_x0 = pred_probs_E / torch.sum(pred_probs_E, dim=-1, keepdim=True)
            prob_E_final[t0mask, ...] = prob_E_final_x0[t0mask, ...]

        sampled_s = diffusion_utils.sample_discrete_features(X_t, prob_E_final, node_mask=node_mask, test=True)

        # 去掉对角线
        diag_mask = torch.eye(n)  # 创建一个单位矩阵
        diag_mask = ~diag_mask.type_as(pred_probs_E).bool()  # 单位阵的类型=E，并且对角False，其余True
        diag_mask = diag_mask.unsqueeze(0)  # 掩码矩阵的维度与E一致
        sampled_s.E = sampled_s.E * diag_mask

        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        return E_s

    @torch.no_grad()
    def sample_zt(self, noisy_data):

        """Samples from zs ~ p(zs | zt). Only used during sampling.
                  if last_step, return the graph prediction as well"""

        node_mask = noisy_data['node_mask']
        X_t = noisy_data['X_t']
        trueE = noisy_data['trueE']
        y_t = noisy_data['y_t']

        # Neural net predictions
        batch, ALLedge_index = self.noisy_data2data_dense(noisy_data, all_edge=False)
        time_tensor_tonode = noisy_data['t'][batch.batch].squeeze(-1)
        torch.set_grad_enabled(False)
        pred = self.model(batch.x, batch.edge_attr, time_tensor_tonode, batch.edge_index, ALLedge_index, node_mask, noisy_data)
        pred = self.pred_shape2data(pred, noisy_data)
        pred1 = pred.mask(node_mask)
        E_s1 = self.given_pred_to_edge(pred1, noisy_data, node_mask, predflah2='weight')

        # pred2 = network_preprocess.PlaceHolder(X=X_t, E=trueE, y=pred.y).mask(node_mask)
        # noisy_data['E_t'] = noisy_data['trueE']
        # E_s2 = self.given_pred_to_edge(pred2, noisy_data, node_mask, predflah2='weight')
        out_one_hot = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)
        out_discrete = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=False).type_as(y_t)

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        y_t = torch.hstack((noisy_data['y_t'], noisy_data['t'])).float()
#        pred = self.model(noisy_data['X_t'], noisy_data['E_t'], y_t, node_mask)
        batch, ALLedge_index = self.noisy_data2data_dense(noisy_data, all_edge=False)
        time_tensor = noisy_data['t'].repeat_interleave(self.max_num_nodes * self.max_num_nodes)
        pred = self.model(batch.x, batch.edge_attr, time_tensor, batch.edge_index, ALLedge_index, node_mask)
        pred = self.dense2data(pred, noisy_data)
    #    pred = self.model(noisy_data['X_t'], noisy_data['E_t'], noisy_data['t'], edgeindex, node_mask)
        # Normalize predictions
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # 对edge进行加权和归一化处理
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))  # 512*81*5
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1

        unnormalized_prob_E = weighted_E.sum(dim=-2)

        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5

        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)

        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(X_t, prob_E, node_mask=node_mask)

        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        out_one_hot = network_preprocess.PlaceHolder(X=X_t, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = network_preprocess.PlaceHolder(X=X_t, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    @torch.no_grad()
    def test_step(self, testdata, TrueData=None, show=True, seed=None):
        """
        :param batch_id: int
        """
        X = testdata.x
        n_nodes = torch.tensor(X.shape[0])         # 采样节点数
        batch_size = torch.tensor(1)      # 采样批次（1次）
        n_max = self.max_num_nodes         # 最大节点数
        # to dense X
        dense_X = torch.zeros([1, n_max, X.shape[1]], device=self.device)
        dense_X[0, :X.shape[0], :] = X

        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(X=dense_X, limit_dist=self.limit_dist, node_mask=node_mask, seed=seed)
        E, y = z_T.E, z_T.y
        y = y.to(E.device)
        # Noedge = torch.tensor([1, 0], device=self.device)
        # GENE_ID_list, TF_ID_list = cal_del_TF_edge(testdata.y)
        all_adj = []
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        if show:
            pbar2 = tqdm(range(self.num_timesteps - 1, -1, -1), ncols=100)
        else:
            pbar2 = range(self.num_timesteps - 1, -1, -1)
        for t_int in pbar2:
            t_array = t_int * torch.ones((batch_size, 1)).type_as(E)
            if t_int == 0:
                s_array = t_array
            else:
                s_array = t_array - 1
            s_norm = s_array / self.num_timesteps
            t_norm = t_array / self.num_timesteps

            # Sample z_s
            # t_int = torch.zeros(size=(1, 1), device=self.device).float() + t_int
            # dense_data = self.apply_noise(dense_data1.X, dense_data1.E, dense_data1.y, node_mask, t_int=t_int)
            noisy_data = {'X_t': dense_X, 'trueE': E, 'E_t': E, 'y_t': y, 's': s_norm, 't': t_norm, 't_int': t_array,
                          's_int': s_array, 'node_mask': node_mask}
            noisy_data['y_t'] = noisy_data['t'].float()
            sampled_s, sampled_s2 = self.sample_zt(noisy_data)
            # for TF_ID in TF_ID_list:
            #     for GENE_ID in GENE_ID_list:
            #         sampled_s.E[:, TF_ID, GENE_ID, :] = Noedge
            E, y = sampled_s.E, sampled_s.y
            sampled_s1 = copy.deepcopy(sampled_s).mask(node_mask, collapse=True)
            sampled_s1.E = sampled_s1.E.squeeze(0)  # 删除批次
            sampled_s1.E = sampled_s1.E[:, node_mask.squeeze(0)]
            sampled_s1.E = sampled_s1.E[node_mask.squeeze(0), :]
            if show:
                performance = Evaluation(y_pred=sampled_s1.E[:, :].flatten(),
                                                  y_true=TrueData.flatten())
                pbar2.set_description(f" t = {int(t_array.cpu().numpy()):3.0f} -- AUC:  {performance['AUC']:.4f} -- AUPR:  {performance['AUPR']:.4f} -- AUPRM:  {performance['AUPR_norm']:.4f}")
            adj = sampled_s1.E.cpu().detach().numpy().copy()
            if testdata.y is not None:
                adj = pd.DataFrame(adj, index=testdata.y, columns=testdata.y)
            all_adj.append(adj)

        return sampled_s1.E, all_adj

    @torch.no_grad()
    def test_step_for_photo(self, testdata, TrueData=None, show=True):
        """
        :param batch_id: int
        """
        X = testdata.x
        n_nodes = torch.tensor(X.shape[0])         # 采样节点数
        batch_size = torch.tensor(1)      # 采样批次（1次）
        n_max = self.max_num_nodes         # 最大节点数
        # to dense X
        dense_X = torch.zeros([1, n_max, X.shape[1]], device=self.device)
        dense_X[0, :X.shape[0], :] = X

        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(X=dense_X, limit_dist=self.limit_dist, node_mask=node_mask)
        E, y = z_T.E, z_T.y
        y = y.to(E.device)
        if show:
            truelabel = TrueData.E

        all_photo_data = []
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        pbar2 = tqdm(range(self.num_timesteps - 1, -1, -1), ncols=100)
        for t_int in pbar2:
            t_array = t_int * torch.ones((batch_size, 1)).type_as(E)
            if t_int == 0:
                s_array = t_array
            else:
                s_array = t_array - 1
            s_norm = s_array / self.num_timesteps
            t_norm = t_array / self.num_timesteps

            # Sample z_s
            # t_int = torch.zeros(size=(1, 1), device=self.device).float() + t_int
            # dense_data = self.apply_noise(dense_data1.X, dense_data1.E, dense_data1.y, node_mask, t_int=t_int)
            noisy_data = {'X_t': dense_X, 'trueE': E, 'E_t': E, 'y_t': y, 's': s_norm, 't': t_norm, 't_int': t_array,
                          's_int': s_array, 'node_mask': node_mask}
            noisy_data['y_t'] = noisy_data['t'].float()
            sampled_s, sampled_s2, photo_data = self.sample_zt_for_photo(noisy_data)
            E, y = sampled_s.E, sampled_s.y
            sampled_s1 = copy.deepcopy(sampled_s).mask(node_mask, collapse=True)
            sampled_s1.E = sampled_s1.E.squeeze(0)  # 删除批次
            sampled_s1.E = sampled_s1.E[:, node_mask.squeeze(0)]
            sampled_s1.E = sampled_s1.E[node_mask.squeeze(0), :]
            if show:
                performance = Evaluation(y_pred=sampled_s1.E[:, :].reshape(-1, 1),
                                                  y_true=truelabel[0, :, :, 1].reshape(-1, 1))
                pbar2.set_description(f" t = {int(t_array.cpu().numpy()):3.0f} -- AUC:  {performance['AUC']:.4f} -- AUPR:  {performance['AUPR']:.4f} ")
            if (t_array.cpu().numpy() % 100 == 0) | (t_array.cpu().numpy() == 0):
                all_photo_data.append(photo_data)


        return sampled_s1.E, all_photo_data

    @torch.no_grad()
    def sample_zt_for_photo(self, noisy_data):

        """Samples from zs ~ p(zs | zt). Only used during sampling.
                  if last_step, return the graph prediction as well"""

        node_mask = noisy_data['node_mask']
        X_t = noisy_data['X_t']
        trueE = noisy_data['trueE']
        y_t = noisy_data['y_t']

        # Neural net predictions
        batch, ALLedge_index = self.noisy_data2data_dense(noisy_data, all_edge=False)
        time_tensor_tonode = noisy_data['t'][batch.batch].squeeze(-1)
        torch.set_grad_enabled(False)
        pred = self.model(batch.x, batch.edge_attr, time_tensor_tonode, batch.edge_index, ALLedge_index, node_mask, noisy_data)
        pred = self.pred_shape2data(pred, noisy_data)
        pred1 = pred.mask(node_mask)
        E_s1 = self.given_pred_to_edge(pred1, noisy_data, node_mask, predflah2='weight')
        pred1E = F.softmax(pred1.E, dim=-1)
        Et = pred1E.cpu().numpy()[0, 0:51, 0:51, 1]
        Et_mask = E_s1.cpu().numpy()[0, 0:51, 0:51, 1]
        out_one_hot = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)
        out_discrete = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)
        photo_data = {'Et': Et, 'Et_mask': Et_mask, 'X': pred.X.cpu().numpy(), 'adj': out_one_hot}
        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=False).type_as(y_t), photo_data