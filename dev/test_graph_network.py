
import torch
import pathlib
from base.wp_model import ProteinSegmenter2

import sys, os
root_path = os.path.join(
    os.path.dirname(__file__), os.pardir)
sys.path.insert(0, root_path)
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch_geometric
from base.PDB_MBS_preprocessing import get_dataset
from base.wp_model import ProteinSegmenter2
from base.input import get_random_train_test_proteins
from base.evaluation import evaluate
from base.monitor import Monitor
import warnings
warnings.filterwarnings('ignore')

print('----- debug -------')

#rilanciare lo script come era prima, per individuare il bug (quello che calcola il numero dei vicino)

# carica il modello
net = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)
model_path = '../metadata/model_0.01_50_2_5_0.pth'
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

#structures_path = pathlib.Path('../../alphafold_staphylococcus')
#input_structures = os.listdir(str(structures_path))

# dato di esempio
_, test_pdb, db_sites = get_random_train_test_proteins()

test_pdb = test_pdb[:5]

test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

#res_df = pd.DataFrame()

#res_df.to_csv('MasterOfMetal2_pred.txt', index=False)

GRAPH_DATASET = get_dataset(test_pdb, db_sites) # prende in input la site list

print(GRAPH_DATASET)

TEST_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in test_pdb if pdb in list(GRAPH_DATASET.keys())]
print('----- debug -------')
for pdb_structure in TEST_GRAPH_DATASET:
    pred = net(pdb_structure)
    pred = torch.softmax(pred, dim=1)
    #print(pred)

    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)
    Y2_idxs = (pdb_structure.y.argmax(dim=1) == 2).nonzero().view(-1)

    #- filtrare quelle sole
    adj = torch_geometric.utils.to_dense_adj(pdb_structure.edge_index)[0]
    sub_adj = adj[:, pred2_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    #print(n_vicini)
    #print(n_vicini.size())
    #sys.exit()

    pred_res = [pdb_structure.residues[j_idx]+f'|{int(n_vicini[j_idx])}|{np.round(pred[j_idx, 2].data.item(), 4)}' for j_idx in pred2_idxs if n_vicini[j_idx] > 2]
    #pred_res = [pdb_structure.residues[j_idx] + f'|{int(n_vicini[j_idx])}' for j_idx in pred2_idxs]


    target_res = [pdb_structure.residues[j_idx] for j_idx in Y2_idxs]
    print('///////',pdb_structure.name, '///////')
    print("Target")
    print(target_res)
    print("Prediction")
    print(pred_res)

# salva la predizione
