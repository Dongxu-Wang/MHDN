import numpy as np
import torch
from torch import nn
from .nn_utils import BasicConv, batched_index_select, act_layer
from .edge import DenseDilatedKnnGraph
import torch.nn.functional as F
from timm.models.layers import DropPath
from .pos_embed import get_2d_relative_pos_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(hidden_features),
            # nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            # nn.BatchNorm2d(in_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        x = input[0]
        graph = input[1]
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x, graph
    
class GAL(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GAL, self).__init__()
        self.W = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False)

        self.a = nn.Conv2d(2*in_channels, 1, 1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU(0.2)
        self.act_layer = act_layer(act)


    def forward(self, x, edge_index, y=None):
        wh = self.W(x)
        whi = batched_index_select(wh, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            whj = batched_index_select(wh, edge_index[0])

        Eij = self.relu(self.a(torch.cat([whi, whj], dim=1)))

        Aij = self.softmax(Eij)
        Aij = F.dropout(Aij, 0.2, training=self.training)


        h_j = torch.mul(whj, Aij)

        h_i = torch.sum(h_j, -1, keepdim=True)

        return h_i

class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='GAL', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'GAL':
            self.gconv = GAL(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, graph, y=None):
        return self.gconv(x, graph, y)
    

class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r

    def forward(self, x, graph):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        x = super(DyGraphConv2d, self).forward(x, graph, y=None)
        return x.reshape(B, -1, H, W).contiguous()
    

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='GAL', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=400, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()

        self.act_layer = act_layer(act)
        self.n = n
        self.r = r
        self.graph_conv = DyGraphConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        x = input[0]
        graph = input[1]
        _tmp = x
        # x = self.fc1(x)
        x = self.graph_conv(x, graph)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return (x, graph)