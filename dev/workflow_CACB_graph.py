"""
Ta taaaa
"""

import torch
from base.wp_model import ProteinSegmenter2
from base import config
import pathlib
import os
import torch_geometric
from torch_geometric.data import Data #
import numpy as np
from base.config import Atom
import sys
import pandas as pd
from base.PDB_MBS_preprocessing import get_dataset, preproces_protein
from base.site_classifier_model import load_site_classifier
from base.input import get_random_train_test_proteins
import warnings
from base.PDB_MBS_preprocessing import get_aminoacid_features


warnings.filterwarnings('ignore')


# Load trainin data
train_pdb, test_pdb, db_sites = get_random_train_test_proteins()
test_pdb = test_pdb[:50]
train_pdb = train_pdb[:100]
train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

# CHIAPPARE TUTTI I CA E CB

input_structures = train_pdb
#input_structures = ['1adu.pdb']
#input_structures = ['1ces.pdb']

#node features: {embedding dell nodo}
#                + {relation feature} => distanze [ca1-ca2, cb1-ca2, ca1-cb2, (angolo diedro)?]

# se res1=GLY -> (cb1-ca2) = 0 ?
# se res2=GLY -> (ca1-cb2) = 0 ?

#message passing neural net

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class RelGCN(torch.nn.Module):

    def __init__(self, num_node_features, num_classes, edge_dim):
        super().__init__()
        HSIZE = 50
        self.conv1 = GATConv(num_node_features, 300, heads=3, concat=False)
        self.conv2 = GATConv(300, 50, heads=3, concat=False)
        self.conv3 = GATConv(50, num_classes, heads=3, concat=False)



    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # edge_weight deve essere 1-dimensional
        #edge_weight = edge_weight[edge_index.tolist()]
        #print(edge_weight)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        #x = F.softmax(x, dim=1)

        return x


class RelGCNddd(torch.nn.Module):

    def __init__(self, num_node_features, num_classes, edge_dim):
        super().__init__()
        HSIZE = 50
        self.conv1 = GCNConv(num_node_features, 300)
        self.conv2 = GCNConv(300, 50)
        self.conv3 = GCNConv(50, num_classes)



    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # edge_weight deve essere 1-dimensional
        #edge_weight = edge_weight[edge_index.tolist()]
        #print(edge_weight)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        #x = F.softmax(x, dim=1)

        return x


graphNet = RelGCN(1024, 3, edge_dim=4)

optim = torch.optim.Adam(params=graphNet.parameters(), lr=0.001)

for epoch in range(30):
    tot_loss = 0.
    avg_acc0, avg_acc1, avg_acc2 = 0., 0., 0.
    counter = 0
    for j, t_pdb in enumerate(input_structures):

        pathT5 = config.ProtT5_EMBEDDINGS_PATH.joinpath(f'{t_pdb[:4]}.protT5.npy')
        #print("=="*20,os.path.exists(pathT5))
        if os.path.exists(pathT5):
            _features = torch.from_numpy(np.load(pathT5)[0,1:])
            #print(_features.shape)

        else:
            continue
        #print(xx)
        #print(xx.shape)
        #sys.exit()


        optim.zero_grad()

        DFS_CA = get_dataset([t_pdb], atom=Atom.CA, sites=train_sites, dataframe_only=True)
        DFS_CB = get_dataset([t_pdb], atom=Atom.CB, sites=train_sites, dataframe_only=True)

        if len(DFS_CA) == 0:
            print("i) Problem with ", t_pdb)
            continue

        df_CA = DFS_CA[t_pdb]

        #print("---", len(df_CA))
        df_CA.reset_index(inplace=True)
        idxs_GLY = df_CA[df_CA['residue_name'] == 'GLY'].index.values
        idxs_no_GLY = df_CA[df_CA['residue_name'] != 'GLY'].index.values

        df_CA = df_CA[df_CA['residue_name'] != 'GLY']
        df_CB = DFS_CB[t_pdb]



        if len(df_CA)!=len(df_CB):
            print("ii) Problem with ",t_pdb)
            continue

        # prende la colonna target del df
        Y_A = torch.from_numpy(df_CA['target'].to_numpy())
        C0_idxs = (Y_A == 0).nonzero().view(-1)
        C1_idxs = (Y_A == 1).nonzero().view(-1)
        C2_idxs = (Y_A == 2).nonzero().view(-1)

        # la trasforma in onehot
        target_CA = torch.zeros(len(Y_A), 3)
        for i, v in enumerate(Y_A):
            target_CA[i, v] = 1.


        coord_CA = torch.from_numpy(df_CA[['x', 'y', 'z']].to_numpy())
        coord_CB = torch.from_numpy(df_CB[['x', 'y', 'z']].to_numpy())

        #adj_matrix please!
        real_distances_CA = torch.sqrt(
            ((coord_CA[:, None, :] - coord_CA[None, :, :]) ** 2).sum(-1))

        connection_mask = (real_distances_CA <= 10).float()

        #print(real_distances_CA)
        #print(connection_mask.sum())

        edge_index = connection_mask.nonzero().t().contiguous().long()

        CA1_CB2 = ((coord_CA[edge_index[0]] - coord_CB[edge_index[1]])**2).sum(-1).sqrt()
        CB1_CA2 = ((coord_CB[edge_index[0]] - coord_CA[edge_index[1]])**2).sum(-1).sqrt()
        CB1_CB2 = ((coord_CB[edge_index[0]] - coord_CB[edge_index[1]]) ** 2).sum(-1).sqrt()
        CA1_CA2 = ((coord_CA[edge_index[0]] - coord_CA[edge_index[1]]) ** 2).sum(-1).sqrt()

        #print(CA1_CA2)
        #print(CB1_CB2)
        #print(CA1_CB2)
        #print(CB1_CA2)

        edge_attr = torch.stack((CA1_CA2, CB1_CB2, CA1_CB2, CB1_CA2), dim=0)

        residues_CA = df_CA['residue_name'].tolist()
        features = get_aminoacid_features(residues_CA, df_CA['Phi'].tolist(), df_CA['Psi'].tolist())
        features = _features[idxs_no_GLY]

        #- create graph object_
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr.t(), y=target_CA)

        pred = graphNet(data)

        loss0 = ((pred[C0_idxs] - data.y[C0_idxs])**2).sum(-1).sqrt().mean()
        loss1 = ((pred[C1_idxs] - data.y[C1_idxs]) ** 2).sum(-1).sqrt().mean()
        loss2 = ((pred[C2_idxs] - data.y[C2_idxs]) ** 2).sum(-1).sqrt().mean()
        loss = loss0 + loss1 + loss2
        loss.backward()
        optim.step()

        prdC2_idxs = (pred.argmax(dim=1) == 2).float().nonzero().view(-1)


        with torch.no_grad():
            tot_loss += loss
            acc0 = (pred[C0_idxs].argmax(dim=1) == 0).float().mean()
            acc1 = (pred[C1_idxs].argmax(dim=1) == 1).float().mean()
            acc2 = (pred[C2_idxs].argmax(dim=1) == 2).float().mean()
            avg_acc0 += acc0
            avg_acc1 += acc1
            avg_acc2 += acc2
            counter+=1

    print(pred[C2_idxs])
    print(pred[prdC2_idxs])

    #print(loss.data, acc0, acc1, acc2)
    print(f"Epoch {epoch} Loss {tot_loss.data} acc0 {avg_acc0/counter} acc1 {avg_acc1/counter} acc2 {avg_acc2/counter}")


