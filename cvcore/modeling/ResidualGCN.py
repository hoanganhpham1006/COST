import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        # nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        nn.init.xavier_uniform_(self.weight.data)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ResidualGCN(nn.Module):
    def __init__(self, nhid, dropout):
        super(ResidualGCN, self).__init__()

        self.gcn_block = GraphConvolution(nhid, nhid)
        self.linear = nn.Linear(nhid, nhid, bias=False)
        self.dropout = dropout

    def forward(self, x, adj, block_index):
        gcn_layer = self.gcn_block(x, adj)
        gcn_layer = F.elu(gcn_layer)
        gcn_layer = F.dropout(gcn_layer, self.dropout, training=self.training)
        x = F.elu(x + self.linear(gcn_layer))
        return x