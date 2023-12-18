import torch
import torch_geometric
from torch_geometric.nn import GATConv # NNConv
import torch.nn.functional as F


class RelationalGraphNN(torch.nn.Module):

    def __init__(self, num_node_features, num_classes):
        super().__init__()
        HSIZE = 50
        self.conv1 = GATConv(num_node_features, HSIZE)
        self.conv2 = GATConv(HSIZE, num_classes)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # edge_weight deve essere 1-dimensional
        edge_weight = edge_weight[edge_index.tolist()]
        #print(edge_weight)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)


        return x


class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=5, out_features=3)

    def forward(self, x):
        return self.fc(x)


nn = NN()

X = torch.randn(3,4,5)
print(X)
print(nn(X))