import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
from torch.nn import functional as F
import torch.nn.init as init

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


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train=0.1):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_E, nosoft_pred_E, true_E, CEloss=False):
        """ Compute train metrics
        masked_pred_E : tensor -- (bs, n, n, de)
        true_E : tensor -- (bs, n, n, de)
        """
        if CEloss == True:
            masked_pred_E = nosoft_pred_E
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        #loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        if CEloss == True:
            loss_train = F.cross_entropy(flat_pred_E, flat_true_E)
        else:
            loss_train = F.binary_cross_entropy(flat_pred_E, flat_true_E)
        return loss_train

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
        # 使用 Kaiming 初始化方法初始化权重
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置项为零
        if m.bias is not None:
            init.zeros_(m.bias)

    for name, param in m.named_parameters():
        if 'weight' in name:
            if 'attention' in name:  # 初始化attention层的权重
                init.xavier_uniform_(param)
            else:  # 初始化其他层的权重
                init.kaiming_uniform_(param)
        elif 'bias' in name:
            init.zeros_(param)

