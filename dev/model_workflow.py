"""

- model CA (TRAINED)

- model CB (TRAINED)

TO DO
- concatenation (or sum)

- final classification (>> TO TRAIN <<)

Optional:

=> Potremmo aggiungere alla fine anche una rete che classifica se sono siti
in base alla CA-CB distance matrix (insomma la rete addestrata in MoM-v1)

"""
import torch
from base.wp_model import ProteinSegmenter2
import pathlib
import os
import torch_geometric
import numpy as np
from base.config import Atom
import sys
import pandas as pd
from base.PDB_MBS_preprocessing import get_dataset
from base.site_classifier_model import load_site_classifier
from base.input import get_random_train_test_proteins
import warnings
warnings.filterwarnings('ignore')


def get_aminoacid_features2(aminoacids):

    #print("<<<<", aminoacids)

    features = np.zeros((len(aminoacids), 5))

    for j, aa in enumerate(aminoacids):
       # idx = configs.AMINOACIDS[aa]
       # features[j, idx] = 1.
       if aa == 'ASP':
           features[j, 0] = 1.
       elif aa == 'HIS':
           features[j, 1] = 1.
       elif aa == 'CYS':
           features[j, 2] = 1.
       elif aa == 'GLU':
           features[j, 3] = 1.
       else:
           features[j, 4] = 1.

    features = torch.from_numpy(features).float()

    AB = torch.zeros(2 * len(features), 2)
    AB[0:len(features), 0] = 1.
    AB[len(features):, 1] = 1.
    features = torch.cat((features, features), dim=0)
    features = torch.cat((features, AB), dim=1)

    return features


def set_vicini(pred_res_, adj_):

    sub_idxs = np.array([d['index'].item() for d in pred_res_])

    # count neughtbors
    sub_adj = adj_[:, sub_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    # da integrare con il pezzo di sopra
    for d in pred_res_:
        j_idx = d['index']
        d['n_vicini'] = int(n_vicini[j_idx]) - 1 # perchè prima ha contato pure se stesso
        sub_vicini_idxs = sub_adj[j_idx].nonzero().view(-1).numpy()
        d['vicini_idxs'] = sub_idxs[sub_vicini_idxs]


def rimuovi_isolati(pred_res_,  min_n_vicini=1):

    # final
    pred_res_out = []
    bbb = []
    for res_dict in pred_res_:
        if ( res_dict['n_vicini'] > min_n_vicini ): # quindi minimo 2
            pred_res_out.append(f"{res_dict['id']}|{res_dict['n_vicini']}|{res_dict['confidence']}")
            bbb.append(res_dict)


    return pred_res_out, bbb


def predict_site(input_structure, model, threshold=0.7):

    _, pred = model(input_structure)

    pred = torch.softmax(pred, dim=1)

    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    #adj = torch_geometric.utils.to_dense_adj(input_structure.edge_index)[0]

    # predicted binding residues info
    pred_res0 = [{'id':input_structure.residues[j_idx],
                  #'n_vicini': int(n_vicini[j_idx])-1,
                  'confidence':np.round(pred[j_idx, 2].data.item(), 4),
                  'index':j_idx}
                 for j_idx in pred2_idxs ]

    pred_res0 = [d for d in pred_res0 if d['confidence'] >= threshold]

    return pred_res0


# adj matrix as in the Master-of-metals V1 (no batch for now)
def mom_v1_adj(xyz_CACB):
    distances = torch.sqrt((xyz_CACB[:, None, :] - xyz_CACB[None, :, :]) ** 2).sum(-1)
    exp_distances = torch.exp(-distances/15).float()
    return exp_distances


def CACB_distance_matrix(graph_CA, graph_CB, residues):
    xyz_CA = graph_CA.df[graph_CA.df['res_pos_chain'].isin(residues)][['x', 'y', 'z']].to_numpy()
    xyz_CB = graph_CB.df[graph_CB.df['res_pos_chain'].isin(residues)][['x', 'y', 'z']].to_numpy()

    coordinates = np.concatenate((xyz_CA, xyz_CB), axis=0)
    coordinates = torch.from_numpy(coordinates).float()
    distances_CACB = mom_v1_adj(coordinates)
    return distances_CACB


# ------ LOAD TRAINED MODELS --------
# load trained model on CA graph
net_CA = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)
# load trained model on CB graph
net_CB = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)

net_CA_path = '../metadata/model_0.01_50_2_5_0.pth'
net_CB_path = 'grid_search/model_0.01_50_2_5_0.CB.pth_FP0'

net_CA.load_state_dict(torch.load(net_CA_path, map_location=torch.device('cpu')))
net_CB.load_state_dict(torch.load(net_CB_path, map_location=torch.device('cpu')))

# il classificatire di poliedri (ca-cb)
net = load_site_classifier('../metadata/trained_model_F0.pth')



# -------- LOAD DATA ----------------
# Load staphylococcus structures list
structures_path = pathlib.Path('../../alphafold_staphylococcus')
input_structures = os.listdir(str(structures_path))

# Load trainin data
train_pdb, test_pdb, db_sites = get_random_train_test_proteins()
test_pdb = test_pdb[:50]
train_pdb = train_pdb[:50]
train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]


input_structures = train_pdb


# -------- PROCESS DATA -------------
res_df = pd.DataFrame()
n_metallo_prot = 0

input_structures = train_pdb[:5]
input_structures = ['1adu.pdb']

GRAPH_CONFIDENCE = 0.5
SITE_CONFIDENCE = 0.6

GRAPH_CONFIDENCE = 0
SITE_CONFIDENCE = 0.2

for j, t_pdb in enumerate(input_structures):

    print("\n========>",t_pdb)

    #input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=[], path_to_structures=structures_path)
    #input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=[], path_to_structures=structures_path)

    input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=train_sites)
    input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=train_sites)


    if len(input_graph_CA) == 0 or len(input_graph_CB) == 0: # attenzione, le GLY non hanno CB
        continue

    input_graph_CA = input_graph_CA[t_pdb]
    input_graph_CB = input_graph_CB[t_pdb]


    # PREDICO I RESIDUI DI CLASSE 2
    pred_res_CA = predict_site(input_graph_CA, net_CA, threshold=GRAPH_CONFIDENCE)
    pred_res_CB = predict_site(input_graph_CB, net_CB, threshold=GRAPH_CONFIDENCE)

    common_res = []

    # Prendo quelli in comune a CA e CB
    if ((len(pred_res_CA) >= 2) and (len(pred_res_CB) >= 2)):
        for x in pred_res_CA:
            common_res.append(x)

        # PRENDI QUELLI IN COMUNE A CA E CB
        #pred_res_CB2 = [x['id'] for x in pred_res_CB]
        #for x in pred_res_CA:
        #    if x['id'] in pred_res_CB2:
        #        common_res.append(x)

    # Identifico i vicini
    adj_CA = torch_geometric.utils.to_dense_adj(input_graph_CA.edge_index)[0]
    set_vicini(common_res, adj_CA)

    #predittore gcn del mom1
    #per ogni residuo predetto... prendere i vicini e vedere se compongono un sito
    for x in common_res:
        #no, qui le coordinate devono essere quelle dei vicini + il residuo, non di out2
        #print("////////////////",x)

        # la lista dei vicini di quel residuo  ...questa è arte!!!
        residues_to_xyz =  input_graph_CA.df.iloc[x['vicini_idxs']]['res_pos_chain'].tolist()

        distances_CACB = CACB_distance_matrix(input_graph_CA, input_graph_CB, residues_to_xyz)
        """xyz_CA = input_graph_CA.df[ input_graph_CA.df['res_pos_chain'].isin(residues_to_xyz)][['x', 'y', 'z']].to_numpy()
        xyz_CB = input_graph_CB.df[input_graph_CB.df['res_pos_chain'].isin(residues_to_xyz)][['x', 'y', 'z']].to_numpy()

        coordinates = np.concatenate( (xyz_CA, xyz_CB), axis= 0)
        coordinates = torch.from_numpy(coordinates).float()
        distances_CACB = mom_v1_adj(coordinates)"""
        #features = get_aminoacid_features2([x.split('_')[0] for x in out2])
        features = get_aminoacid_features2([x.split('_')[0] for x in residues_to_xyz])
        """AB = torch.zeros(2*len(features), 2)
        AB[0:len(features), 0] = 1.
        AB[len(features):, 1] = 1.
        features = torch.cat((features, features), dim=0)
        features = torch.cat((features, AB), dim=1)"""

        #hidden, net_out = net(potential_mbs.x, potential_mbs.adj.float())

        #print(f'features {features.size()}, distances_CACB {distances_CACB.size()}')
        hidden, net_out = net(features, distances_CACB)

        P_mbs = round(net_out[1].item(), 4)

        x['pred_mbs'] = P_mbs

        print(x['id'], '|=', residues_to_xyz, f'P(site)=', P_mbs)

    #ogni aminoacido predetto con si, appartiene a un sito...
    #outt = [f"{x['id']}|{x['n_vicini']}|{x['confidence']}" for x in common_res if  x['pred_mbs'] > 0.6 ]

    common_res_ok = [x for x in common_res if x['pred_mbs'] > SITE_CONFIDENCE]

    #QUI VANNO TOLTI QUELLI SENZA VICINI
    outt, common_res_ok_grouped = rimuovi_isolati(common_res_ok)

    binding_residues = ';'.join(outt)
    #binding_residues = [x.split('|')[0] for x in outt]

    print("Predicted:",binding_residues)
    t2 = (input_graph_CA.y.argmax(dim=1) == 2).nonzero().view(-1)
    qq=[f"{x.split('|')[0].split('_')[1]}" for x in outt]
    p_query = f"select resid {'+'.join(qq)}"
    print(p_query)

    xx = [input_graph_CA.residues[j] for j in t2]
    print("Target:", xx)
    qq = [x.split('_')[1] for x in xx]
    t_query = f"select resid {'+'.join(qq)}"
    print(t_query)

    if len(outt) > 2:
        res_df = res_df.append({'input_structure':t_pdb, 'n_binding_res': len(outt),
                                'predicted': binding_residues}, ignore_index=True)
        n_metallo_prot += 1
    else:
        res_df = res_df.append({'input_structure': t_pdb, 'n_binding_res': len(outt),
                                'predicted': 'NO_SITES_PREDICTED'}, ignore_index=True)

    print(f'{j}/{len(input_structures)} - {t_pdb} - # metallo prot {n_metallo_prot}')


res_df.to_csv('MasterOfMetal2_pred_14nov.ALL.txt', index=False)


#--- VEDERE COME PERFORMA SUL TRAINING SET ---