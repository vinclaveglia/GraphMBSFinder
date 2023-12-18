import sys
import numpy as np
import pandas as pd
import os
from base import config
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def create_graph_object(features, adj_matrix, node_target=None, df= None, name = "prova"):

    edge_index = adj_matrix.nonzero().t().contiguous().long()

    x = features

    if node_target != None:
        data = Data(x=x, edge_index=edge_index, y=node_target)
    else:
        data = Data(x=x, edge_index=edge_index)

    data["name"] = name
    data['residues'] = df['res_pos_chain'].tolist()

    return data

def get_graph(prot):

    features = np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.pdb.feat'))
    features = torch.from_numpy(features).float()

    distances = np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.pdb.dist'))
    distances = torch.from_numpy(distances).float()

    df = pd.read_csv(os.path.join(config.PROTS_PATH, f'{prot}.pdb.labels'))

    target = torch.tensor(df['target'])
    target_oh = F.one_hot(target, num_classes=3)

    graph = create_graph_object(features, adj_matrix=distances, node_target=target_oh, name=prot, df=df)

    return graph

def get_test_sites():

    zn_folds = pd.read_csv("../metadata/Zn_folds.csv")

    data = zn_folds.iloc[0]['sites'].split(';')

    return data

def get_train_test():

    #zn_folds = pd.read_csv("metadata/Zn_folds.csv")
    zn_folds = pd.read_csv(config.METADATA_PATH.joinpath("Zn_folds.csv"))

    df_test = zn_folds[zn_folds['fold_id']=='fold_0']

    df_train = zn_folds[zn_folds['fold_id'].isin(['fold_1',
                                                  'fold_2',
                                                  'fold_3',
                                                  'fold_4'])]

    test_sites, train_sites = [], []

    for _, row in df_train.iterrows():
        train_sites += row['sites'].split(';')

    for _, row in df_test.iterrows():
        test_sites += row['sites'].split(';')


    return train_sites, test_sites

def get_random_train_test_proteins():

    file_dir = os.path.dirname(__file__)

    all_sites = []
    all_proteins = []

    # legge il file di Beatrice e carica tutti i siti
    zn_folds = pd.read_csv(os.path.join(file_dir, "../metadata/Zn_folds.csv"))

    for _, row in zn_folds.iterrows():
        all_sites += row['sites'].split(';')

    # organizza per proteins
    for site in all_sites:
        pdb_code = site.split("_")[0]
        all_proteins.append(f'{pdb_code}.pdb')
    all_proteins = sorted(list(set(all_proteins)))

    # random train e test proteins
    n_prots = len(all_proteins)
    train_size = int(n_prots*0.8)

    train_proteins = all_proteins[:train_size]
    test_proteins = all_proteins[train_size:]

    return train_proteins, test_proteins, all_sites