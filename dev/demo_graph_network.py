
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
from base.config import Atom
import warnings
warnings.filterwarnings('ignore')

print("yuppiiiiii")

# carica il modello
net = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)

model_path = '../metadata/model_0.01_50_2_5_0.pth'
model_path = 'grid_search/model_0.01_50_2_5_0.CB.pth_FP0'

net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


structures_path = pathlib.Path('../../alphafold_staphylococcus')
input_structures = os.listdir(str(structures_path))

# dato di esempio
#_, test_pdb, db_sites = get_random_train_test_proteins()
#test_pdb = test_pdb[:5]
#test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]


def predict_site(input_structure, model, threshold=0.7):

    pred = model(input_structure)
    pred = torch.softmax(pred, dim=1)
    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    adj = torch_geometric.utils.to_dense_adj(input_structure.edge_index)[0]

    # predicted binding residues info
    pred_res0 = [{'id':input_graph.residues[j_idx],
                  #'n_vicini': int(n_vicini[j_idx])-1,
                  'confidence':np.round(pred[j_idx, 2].data.item(), 4),
                  'index':j_idx}
                 for j_idx in pred2_idxs ]

    # only residues with high confidence
    pred_res1 = [d for d in pred_res0 if d['confidence'] >= threshold]
    sub_idxs = [d['index'] for d in pred_res1]

    # count neughtbors
    sub_adj = adj[:, sub_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    for d in pred_res1:
        j_idx = d['index']
        d['n_vicini'] = int(n_vicini[j_idx])-1

    # final
    pred_res_ = []
    for res_dict in pred_res1:
        if ((res_dict['n_vicini'] > 0) and (res_dict['confidence'] >= threshold)):
            pred_res_.append(
                f"{res_dict['id']}|{res_dict['n_vicini']}|{res_dict['confidence']}")


    return pred_res_



res_df = pd.DataFrame()

n_metallo_prot = 0
for j, t_pdb in enumerate(input_structures):

    # in pratica gli chiediamo di creare un dataset di un singolo componente
    input_graph = get_dataset([t_pdb], atom=Atom.CB, sites=[], path_to_structures=structures_path)
    #salvala su disco come oggetto se non esiste ancora

    #input_graph = input_graph[ list(input_graph.keys())[0] ]
    input_graph = input_graph[t_pdb]

    """
    pred = net(input_graph)
    pred = torch.softmax(pred, dim=1)
    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    #- filtrare quelle sole
    adj = torch_geometric.utils.to_dense_adj(input_graph.edge_index)[0]
    #sub_adj = adj[:, pred2_idxs]
    #n_vicini = sub_adj.sum(dim=1).tolist()

    #pred_res2 = {}

    #pred_res = [input_graph.residues[j_idx] + f'|{int(n_vicini[j_idx])-1}|{np.round(pred[j_idx, 2].data.item(), 4)}'
    # for j_idx in pred2_idxs if (n_vicini[j_idx] > 1)] # tolgo quelli isolati
    #pred_res = [input_graph.residues[j_idx] + f'|{int(n_vicini[j_idx])}' for j_idx in pred2_idxs ]

    pred_res0 = [{'id':input_graph.residues[j_idx],
                  #'n_vicini': int(n_vicini[j_idx])-1,
                  'confidence':np.round(pred[j_idx, 2].data.item(), 4),
                  'index':j_idx}
                 for j_idx in pred2_idxs ]

    pred_res1 = [d for d in pred_res0 if d['confidence'] > 0.7]

    #ATTENZIONE: la conta del numero dei vicini va fatta qui...

    sub_idxs = [d['index'] for d in pred_res1]
    sub_adj = adj[:, sub_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    for d in pred_res1:
        j_idx = d['index']
        d['n_vicini'] = int(n_vicini[j_idx])-1

    pred_res = []
    for res_dict in pred_res1:
        if ((res_dict['n_vicini'] > 0) and (res_dict['confidence'] >= 0.7)):
            pred_res.append(
                f"{res_dict['id']}|{res_dict['n_vicini']}|{res_dict['confidence']}")
    """

    pred_res = predict_site(input_graph, net, threshold=0.9)

    #print(t_pdb + ',' +    ';'.join(pred_res))
    binding_residues = ';'.join(pred_res)
    if len(pred_res) > 2:
        res_df = res_df.append({'input_structure':t_pdb, 'n_binding_res': len(pred_res),
                                'predicted': binding_residues}, ignore_index=True)
        n_metallo_prot += 1
    else:
        res_df = res_df.append({'input_structure': t_pdb, 'n_binding_res': len(pred_res),
                                'predicted': 'NO_SITES_PREDICTED'}, ignore_index=True)

    print(f'{j}/{len(input_structures)} - {t_pdb} - # metallo prot {n_metallo_prot}')


res_df.to_csv('MasterOfMetal2_pred_25ott.CB.txt', index=False)
sys.exit()








GRAPH_DATASET = get_dataset(test_pdb, db_sites) # prende in input la site list

TEST_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in test_pdb if pdb in list(GRAPH_DATASET.keys())]

for pdb_structure in TEST_GRAPH_DATASET:
    print(pdb_structure)
    pred = net(pdb_structure)

    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)
    Y2_idxs = (pdb_structure.y.argmax(dim=1) == 2).nonzero().view(-1)

    #- filtrare quelle sole
    adj = torch_geometric.utils.to_dense_adj(pdb_structure.edge_index)[0]
    sub_adj = adj[:, pred2_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    #print(n_vicini)
    #print(n_vicini.size())
    #sys.exit()

    pred_res = [pdb_structure.residues[j_idx]+f'|{n_vicini[j_idx]}' for j_idx in pred2_idxs if n_vicini[j_idx] > 2]
    target_res = [pdb_structure.residues[j_idx] for j_idx in Y2_idxs]
    print('///////',pdb_structure.name, '///////')
    print("Target")
    print(target_res)
    print("Prediction")
    print(pred_res)

# salva la predizione