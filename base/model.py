
import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class SecondGCN(torch.nn.Module):

    def __init__(self, num_node_features, num_classes):
        super().__init__()
        HSIZE = 50
        self.conv1 = GCNConv(num_node_features, HSIZE)
        self.conv2 = GCNConv(HSIZE, HSIZE)
        self.conv3 = GCNConv(HSIZE, num_classes)



    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # edge_weight deve essere 1-dimensional
        edge_weight = edge_weight[edge_index.tolist()]
        #print(edge_weight)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)

        #x = F.softmax(x, dim=1)

        return x



class ProteinSegmenter2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        HSIZE = 30
        self.conv1 = GCNConv(num_node_features, HSIZE)
        self.conv2 = GCNConv(HSIZE, HSIZE)
        self.conv3 = GCNConv(HSIZE, num_classes)
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
        x = F.dropout(x, training=self.training)
        #x = F.layer_norm(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)

        #x = self.conv4(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)

        #x = self.conv5(x, edge_index)

        #x = torch.sigmoid(x)
        #return F.log_softmax(x, dim=1)
        return x