import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer

"""
from discrete.models.GraphTransformer.graph_transformer_layer import GraphTransformerLayer
from discrete.models.GraphTransformer.mlp_readout_layer import MLPReadout


class GraphTransformerNet(nn.Module):
    def __init__(self,
                 num_features,
                 channels,
                 max_time=1000,
                 gamma=1,
                 device='cuda:0',
                 learned_sinusoidal_cond=False,
                 random_fourier_features=False,
                 learned_sinusoidal_dim=16,
                 self_condition=False):
        super().__init__()
        self.max_time = max_time
        self.gamma = gamma
        self.channels = channels
        self.self_condition = self_condition
        self.type = 'dot'

        in_dim_node = num_features  # node_dim (feat is an integer)
        hidden_dim = channels
        out_dim = channels
        num_heads = 5
        in_feat_dropout = channels
        dropout = channels
        n_layers = 3
        n_classes = 2
        self.readout = 128
        self.layer_norm = True
        self.batch_norm = True
        self.residual = True
        self.dropout = dropout
        self.device = device
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = 512
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                                           dropout, self.layer_norm, self.batch_norm, self.residual) for
                                     _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        time_dim_out = int(channels / 2)
        self.time_dim_out = time_dim_out
        self.time_mlp2 = nn.Embedding(channels, int(channels / 2))
        self.time_mlp3 = nn.Linear(in_features=time_dim_out, out_features=time_dim_out)
        self.MLP_layer = MLPReadout(out_dim * 2 + time_dim_out, n_classes)

    def forward(self, x, edge_attr, t_step, edge_index, ALLedge_index, node_mask):
        graph = dgl.DGLGraph()
        graph.add_edges(edge_index[0], edge_index[0])



        # input embedding
        x = self.embedding_h(x)
        x = self.in_feat_dropout(x)

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, x)

        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    def forward(self, x, edge_attr, t_step, edge_index, ALLedge_index, node_mask):
        #    x = self.embedding(x)
        #    for layer in self.layers:
        #        x = F.relu(layer(x, edge_index))

        for layer in self.layers:
            x = layer(x, edge_index)

        # 嵌入时间
        t_emb = self.compute_emb(t_step)
        t_emb = self.time_mlp3(t_emb)

        x_TF = self.fc_tf(x)
        x_Target = self.fc_target(x)

        # 将边连接成对应的节点特征，并计算边的权重
        edge_features = self.encode(x_TF[ALLedge_index[0]], x_Target[ALLedge_index[1]], type=self.type)
        if self.type == 'dot':
            edge_weights = self.fc_grn_dot(edge_features)
        else:
            edge_weights = self.fc_grn_mlp(edge_features)

        edge_weights = torch.cat([edge_weights, t_emb], dim=1)
        edge_weights = self.fc_out(edge_weights)

        return edge_weights