from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import math

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body'):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)

    def forward(self, x):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)                       ## Unlike the conv layer, it use a only MLP layer without graph conv using the adjacency matrix

        z = x+y

        return z

# class GraphResBlock(torch.nn.Module):
#     """
#     Graph Residual Block similar to the Bottleneck Residual Block in ResNet
#     """
#     def __init__(self, in_channels, out_channels, mesh_type='body'):
#         super(GraphResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = GraphConvolution(self.in_channels, self.out_channels, mesh_type)
#         print('Use BertLayerNorm and GeLU in GraphResBlock')
#         self.norm = BertLayerNorm(self.out_channels)
#     def forward(self, x):
#         y = self.conv(x)
#         y = self.norm(y)
#         y = gelu(y)
#         z = x+y
#         return z

class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True):
        super(GraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features

        # if mesh=='body':
        #     adj_indices = torch.load('../src/modeling/data/smpl_431_adjmat_indices.pt')
        #     adj_mat_value = torch.load('../src/modeling/data/smpl_431_adjmat_values.pt')
        #     adj_mat_size = torch.load('../src/modeling/data/smpl_431_adjmat_size.pt')
        # elif mesh=='hand':
        #     # adj_indices = torch.load('../modeling/data/mano_195_adjmat_indices.pt')
        #     # adj_mat_value = torch.load('../modeling/data/mano_195_adjmat_values.pt')
        #     # adj_mat_size = torch.load('../modeling/data/mano_195_adjmat_size.pt')
        adj_indices = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20],
                        [1, 5, 9, 13, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 0, 10, 9, 11, 10, 12, 11, 0, 14, 13, 15, 14, 16, 15, 0, 18, 17, 19, 18, 20, 19]])
        adj_mat_value = torch.ones(40)
        adj_mat_size = (21, 21)
        self.adjmat = torch.sparse_coo_tensor(adj_indices, adj_mat_value, size=adj_mat_size).to(device)

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)       ## Adj matrix: 21 x 21 / x: 32 x 21 x 32 / weight: (32 x 32 -> input_feature D x output_feature D)
                                                                ## support: adjust the dimension of input through the MLP layer 
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(self.adjmat, support))       ## Multiply the sparse matrix because of memory space
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'