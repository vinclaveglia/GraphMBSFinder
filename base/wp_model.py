
import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        input = F.dropout(input, self.dropout, self.training)

        D = torch.diag((adj > 0.).int().sum(dim=1))
        #print(D)

        support = torch.mm(input, self.weight)
        I = torch.zeros_like(adj).fill_diagonal_(1.)
        #print(adj)
        #print(  torch.matmul( torch.matmul((D ** 0.5), (adj + I)), (D ** 0.5))  )

        DD = torch.diag(D.diagonal() ** -0.5)

        A_tilda = DD @ (adj + I) @ DD

        #A_tilda = adj
        output = torch.spmm(A_tilda, support)

        return output + self.bias


class ProteinSegmenter(nn.Module):

    def __init__(self, nfeat, n_h, n_out, dropout):

        super(ProteinSegmenter, self).__init__()
        HH = 50
        self.gc1 = GraphConvolution(nfeat, HH, dropout)

        self.gc2 = GraphConvolution(HH, HH, dropout)

        #self.gc3 = GraphConvolution(HH, HH, dropout)

        self.gc_out = GraphConvolution(HH, n_out, dropout)

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))

        x = F.relu(self.gc2(x, adj))

        #x = F.relu(self.gc3(x, adj))

        y = self.gc_out(x, adj)

        #y = torch.sigmoid(y)

        return y


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class ProteinSegmenter2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, layer_size = 30):
        super().__init__()
        HSIZE = layer_size
        self.conv1 = SAGEConv(num_node_features, HSIZE)
        self.conv2 = SAGEConv(HSIZE, HSIZE)
        self.conv3 = SAGEConv(HSIZE, HSIZE)
        self.conv4 = SAGEConv(HSIZE, num_classes)
        #self.conv4 = SAGEConv(HSIZE, HSIZE)
        #self.conv5 = SAGEConv(HSIZE, num_classes)
        #self.conv3 = GATConv(HSIZE, num_classes)
        #self.reset()
        #self.layernorm = nn.LayerNorm()

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, training=self.training)
        #x = F.layer_norm(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, training=self.training)

        hidden = self.conv3(x, edge_index)
        hidden = F.elu(hidden)
        #x = F.dropout(x, training=self.training)

        x4 = self.conv4(hidden, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)

        #x = self.conv5(x, edge_index)

        #x = torch.sigmoid(x)
        #return F.log_softmax(x, dim=1)
        return hidden, x4