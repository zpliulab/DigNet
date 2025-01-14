import math
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from discrete import network_preprocess
from discrete import diffusion_utils
from discrete.models.layers import Xtoy, Etoy, masked_softmax
'''
    Note:
    The `NodeEdgeBlock` used in this implementation is based on an external algorithm.
    For more details, refer to the GitHub repository: [https://github.com/cvignac/DiGress].
    Additionally, this implementation includes modifications to adapt the algorithm to specific requirements.
'''
class Transformer_encoder(nn.Module):
    def __init__(
        self,
        trans_GE: int,
        trans_E: int,
        trans_T: int,
        n_head: int,
        MLP_trans_GE: int = 2048,
        MLP_trans_E: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None
    ) -> None:
        '''
            Transformer Encoder Module

            This module defines a Transformer-based encoder for processing graph-structured data.
            Key features include:
            - Self-Attention: Captures relationships between nodes, edges, and types using a custom NodeEdgeBlock.
            - Feed-Forward Network: Processes the features with multi-layer perceptrons for nodes, edges, and types.
            - Normalization and Dropout: Includes LayerNorm and Dropout for regularization and stability.
            - Modular Design: Uses nn.Sequential for compact and reusable layers.

            Parameters:
            - trans_GE, trans_E, trans_T: Input dimensions for nodes, edges, and types.
            - n_head: Number of attention heads.
            - MLP_trans_GE, MLP_trans_E, dim_ffy: Dimensions for intermediate layers in the feed-forward network.
            - dropout: Dropout rate for regularization.
            - layer_norm_eps: Epsilon for layer normalization stability.
            - device, dtype: Specifies the hardware device and data type.

            Usage:
            - Call the forward method with node, edge, and type features along with a node mask.
            - Outputs processed node, edge, and type features.
        '''

        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(trans_GE, trans_E, trans_T, n_head, **kw)

        self.MLP_GE = nn.Sequential(
            Linear(trans_GE, MLP_trans_GE, **kw),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(MLP_trans_GE, trans_GE, **kw)
        )

        self.MLP_E = nn.Sequential(
            Linear(trans_E, MLP_trans_E, **kw),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(MLP_trans_E, trans_E, **kw)
        )

        self.MLP_T = nn.Sequential(
            Linear(trans_T, dim_ffy, **kw),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(dim_ffy, trans_T, **kw)
        )

        self.Norm_GE1 = LayerNorm(trans_GE, eps=layer_norm_eps, **kw)
        self.Norm_E1 = LayerNorm(trans_E, eps=layer_norm_eps, **kw)
        self.Norm_T1 = LayerNorm(trans_T, eps=layer_norm_eps, **kw)

        self.Norm_GE2 = LayerNorm(trans_GE, eps=layer_norm_eps, **kw)
        self.Norm_E2 = LayerNorm(trans_E, eps=layer_norm_eps, **kw)
        self.Norm_T2 = LayerNorm(trans_T, eps=layer_norm_eps, **kw)

    def forward(self, X, E, t, node_mask):
        # Encoder: 1. self-Attention
        newX, newE, new_y = self.self_attn(X, E, t, node_mask=node_mask)

        # Encoder: 2. res_add & normalize layer
        X = self.Norm_GE1(X + newX)
        E = self.Norm_E1(E + newE)
        t = self.Norm_T1(t + new_y)

        # Encoder: 3. Feed Forward (Linear + ReLU + Linear)
        X = self.Norm_GE2(X + self.MLP_GE(X))  # Encoder: 4. add & normalize layer
        E = self.Norm_E2(E + self.MLP_E(E))
        t = self.Norm_T2(t + self.MLP_T(t))

        return X, E, t



class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(self, trans_GE, trans_E, trans_T, n_head, **kwargs):
        super().__init__()
        self.trans_GE = trans_GE
        self.trans_E = trans_E
        self.trans_T = trans_T
        self.df = int(trans_GE / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(trans_GE, trans_GE)
        self.k = Linear(trans_GE, trans_GE)
        self.v = Linear(trans_GE, trans_GE)

        # FiLM E to X
        self.e_add = Linear(trans_E, trans_GE)
        self.e_mul = Linear(trans_E, trans_GE)

        # FiLM y to E
        self.y_e_mul = Linear(trans_T, trans_GE)  # Warning: here it's trans_GE and not trans_E
        self.y_e_add = Linear(trans_T, trans_GE)

        # FiLM y to X
        self.y_x_mul = Linear(trans_T, trans_GE)
        self.y_x_add = Linear(trans_T, trans_GE)

        # Process y
        self.y_y = Linear(trans_T, trans_T)
        self.x_y = Xtoy(trans_GE, trans_T)
        self.e_y = Etoy(trans_E, trans_T)

        # Output layers
        self.x_out = Linear(trans_GE, trans_GE)
        self.e_out = Linear(trans_GE, trans_E)
        self.y_out = nn.Sequential(nn.Linear(trans_T, trans_T), nn.ReLU(), nn.Linear(trans_T, trans_T))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask  # (bs, n, trans_GE)
        K = self.k(X) * x_mask  # (bs, n, trans_GE)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with trans_GE = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, trans_GE
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, trans_GE
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df) FiLM(E,Y) = Y .* E1 + Y + E2, where E1 = liner1(E), E2 = liner2(E)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, trans_GE
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, trans_E
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + ( ye2 + 1) * newE  # (bs, n, n, n_head, df) FiLM(y,newE) = newE .* y1 + newE + y2, where E1 = liner1(E), E2 = liner2(E)

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, trans_E
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask  # bs, n, trans_GE
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, trans_GE

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, trans_T

        return newX, newE, new_y


class DigNetGraphTransformer(nn.Module):
    """
    A graph transformer model for processing dynamic graph-structured data.

    Attributes:
        n_layers (int): Number of transformer layers.
        input_dims (dict): Dimensions of the input features for nodes (X), edges (E), and timesteps (y).
        hidden_mlp_dims (dict): Dimensions of the intermediate layers in the MLP blocks.
        hidden_dims (dict): Dimensions of hidden states in the transformer encoder.
        output_dims (dict): Dimensions of the output features for nodes (X), edges (E), and timesteps (y).
        act_fn_in (nn.Module): Activation function used in the input MLP layers.
        act_fn_out (nn.Module): Activation function used in the output MLP layers.
    """


    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        # Transformer layer count
        self.n_layers = n_layers

        # Output dimensions
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        # MLP for node features
        self.MLP_GE = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_mlp_dims['X']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']),
            act_fn_in
        )

        # MLP for edge features
        self.MLP_Edge = nn.Sequential(
            nn.Linear(input_dims['E'], hidden_mlp_dims['E']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']),
            act_fn_in
        )

        # MLP for timestep features
        self.MLP_timestep = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_mlp_dims['y']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']),
            act_fn_in
        )

        # Transformer encoder layers
        self.Transformer_encoder_M = nn.ModuleList([
            Transformer_encoder(
                trans_GE=hidden_dims['dx'],
                trans_E=hidden_dims['de'],
                trans_T=hidden_dims['dy'],
                n_head=hidden_dims['n_head'],
                MLP_trans_GE=hidden_dims['dim_ffX'],
                MLP_trans_E=hidden_dims['dim_ffE']
            ) for _ in range(n_layers)
        ])

        # Output MLP for node features
        self.MLP_GE_O = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], output_dims['X'])
        )

        # Output MLP for edge features
        self.MLP_Edge_O = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['E'], output_dims['E'])
        )

        # Output MLP for timestep features
        self.MLP_timestep_O = nn.Sequential(
            nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['y'], output_dims['y'])
        )

    def forward(self, x, edge_attr, t_step, edge_index, ALLedge_index, node_mask, noisy_data):
        # Extract input data from noisy_data
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']

        # Batch size and number of nodes
        bs, n = X.shape[0], X.shape[1]

        # Create diagonal mask to prevent self-loops
        diag_mask = torch.eye(n, dtype=torch.bool, device=E.device).logical_not()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Initial feature transformation using MLPs
        after_in = network_preprocess.PlaceHolder(
            X=self.MLP_GE(X),
            E=self.MLP_Edge(E),
            y=self.MLP_timestep(y)
        ).mask(node_mask)

        # Update features after masking
        X, E, t = after_in.X, after_in.E, after_in.y

        # Pass through transformer encoder layers
        for layer in self.Transformer_encoder_M:
            X, E, t = layer(X, E, t, node_mask)

        # Output transformation using MLPs
        X = self.MLP_GE_O(X)
        E = self.MLP_Edge_O(E)
        t = self.MLP_timestep_O(t)

        # Apply diagonal mask to edge features
        E = E * diag_mask

        # Return processed features
        return network_preprocess.PlaceHolder(X=X, E=E, y=t).mask(node_mask)
