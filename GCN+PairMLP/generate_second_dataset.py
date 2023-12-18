"""

- model CA (TRAINED)

- model CB (TRAINED)

TO DO
- A partire dai residui predetti come binding-role-2, si costruisce un nuovo grafo prendendo sia i CA che i CB.
Una GCN viene addestrata a predirre quali sono i residui che compongono il sito.
In questo caso i grafi sono molto più piccoli... dovrebbe/potrebbe funzionare


"""
import sys
sys.path.insert(0, '..')
import torch
from base.wp_model import ProteinSegmenter2
from base.model import SecondGCN
import pathlib
import os
import torch_geometric
import numpy as np
import torch
from torch_geometric.data import Data
from base.config import Atom
import pandas as pd
import random
from base.PDB_MBS_preprocessing import get_dataset
from base.site_classifier_model import load_site_classifier
from base.input import get_random_train_test_proteins
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def add_cacb_features(features):

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
        if ( res_dict['n_vicini'] > min_n_vicini ): # quinti minimo 2
            pred_res_out.append(f"{res_dict['id']}|{res_dict['n_vicini']}|{res_dict['confidence']}")
            bbb.append(res_dict)


    return pred_res_out, bbb


def predict_site(input_structure, model, threshold=0.7, hidden=False):

    hidden, pred = model(input_structure)
    #print(pred.grad)
    #sys.exit()

    pred = torch.softmax(pred, dim=1)

    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    #adj = torch_geometric.utils.to_dense_adj(input_structure.edge_index)[0]

    # predicted binding residues info
    pred_res0 = [{'id':input_structure.residues[j_idx],
                  #'n_vicini': int(n_vicini[j_idx])-1,
                  #'embedding':hidden[j_idx],
                  'confidence':np.round(pred[j_idx, 2].item(), 4),
                  'index':j_idx.item()}
                 for j_idx in pred2_idxs ]

    pred_res0 = [d for d in pred_res0 if d['confidence'] >= threshold]

    idxs_ = [ x['index'] for x in pred_res0]

    if hidden is False:
        return pred_res0
    else:
        return pred_res0, hidden[idxs_]


# adj matrix as in the Master-of-metals V1 (no batch for now)
def mom_v1_adj(xyz_CACB):
    distances = torch.sqrt((xyz_CACB[:, None, :] - xyz_CACB[None, :, :]) ** 2).sum(-1)
    #exp_distances = torch.exp(-distances/15).float()
    return distances


def CACB_distance_matrix(graph_CA, graph_CB, residues):
    xyz_CA = graph_CA.df[graph_CA.df['res_pos_chain'].isin(residues)][['x', 'y', 'z']].to_numpy()
    xyz_CB = graph_CB.df[graph_CB.df['res_pos_chain'].isin(residues)][['x', 'y', 'z']].to_numpy()

    coordinates = np.concatenate((xyz_CA, xyz_CB), axis=0)
    coordinates = torch.from_numpy(coordinates).float()
    distances_CACB = mom_v1_adj(coordinates)
    return distances_CACB


def cross_concatenate(X):
    L = len(X)
    x_a = X[None, :, :].expand(L, -1, -1)
    x_b = X[:, None, :].expand(-1, L, -1)
    x_c = torch.cat((x_a, x_b), dim=2)
    return x_c


def cross_distance(X1, X2):
    if len(X1) != len(X2):
        print(t_pdb)
    assert len(X1) == len(X2)

    xdist = ((X1[None, :, :] - X2[:, None, :]) ** 2).sum(-1).sqrt()

    return xdist


# ------ LOAD TRAINED MODELS --------
# load trained model on CA graph
net_CA = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)
# load trained model on CB graph
net_CB = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)

net_CA_path = '../metadata/model_0.01_50_2_5_0.pth'
net_CB_path = '../grid_search/model_0.01_50_2_5_0.CB.pth_FP0'

net_CA.load_state_dict(torch.load(net_CA_path, map_location=torch.device('cpu')))
net_CB.load_state_dict(torch.load(net_CB_path, map_location=torch.device('cpu')))

net_CA.to(device)
net_CB.to(device)

# il classificatire di poliedri (ca-cb)
net = load_site_classifier('../metadata/trained_model_F0.pth')



# -------- LOAD DATA ----------------
# Load staphylococcus structures list
structures_path = pathlib.Path('../../../alphafold_staphylococcus')
input_structures = os.listdir(str(structures_path))

# Load trainin data
train_pdb, test_pdb, db_sites = get_random_train_test_proteins()
#test_pdb = test_pdb[:50]
#train_pdb = train_pdb[:50]
train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

#input_structures = train_pdb[:50]
input_structures = test_pdb

# -------- PROCESS DATA -------------
res_df = pd.DataFrame()
n_metallo_prot = 0

#input_structures = ['1adu.pdb']

GRAPH_CONFIDENCE = 0.1

#optim_graph = torch.optim.Adam(params=net_CA.parameters(), lr=0.001)

#loss_fn = torch.nn.CrossEntropyLoss()

df = pd.DataFrame(columns=['pdb', 'cluster', 'x1', 'x2', 'rel', 'T1', 'T2'])

for j, t_pdb in enumerate(input_structures):
    print(f"========>{j}/{len(input_structures)} - {t_pdb}")

    input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=test_sites)
    input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=test_sites)

    if len(input_graph_CA) == 0 or len(input_graph_CB) == 0:
        continue

    input_graph_CA = input_graph_CA[t_pdb]
    input_graph_CB = input_graph_CB[t_pdb]

    # PREDICO I RESIDUI DI CLASSE 2
    # with torch.no_grad():
    input_graph_CA.to(device)
    input_graph_CB.to(device)
    pred_res_CA, hidden_CA = predict_site(input_graph_CA, net_CA, threshold=GRAPH_CONFIDENCE, hidden=True)
    pred_res_CB, hidden_CB = predict_site(input_graph_CB, net_CB, threshold=GRAPH_CONFIDENCE, hidden=True)

    common_res = []

    # Prendo quelli in comune a CA e CB
    if ((len(pred_res_CA) >= 2) and (len(pred_res_CB) >= 2)):
        for x in pred_res_CA:
            common_res.append(x)

        # PRENDI QUELLI IN COMUNE A CA E CB
        # pred_res_CB2 = [x['id'] for x in pred_res_CB]
        # for x in pred_res_CA:
        #    if x['id'] in pred_res_CB2:
        #        common_res.append(x)

    common_idxs = [x['index'] for x in common_res]
    common_res_names = [x['id'] for x in common_res]

    df_CA_sub = input_graph_CA.df[input_graph_CA.df['res_pos_chain'].isin(common_res_names)]
    df_CB_sub = input_graph_CB.df[input_graph_CB.df['res_pos_chain'].isin(common_res_names)]

    names = df_CA_sub['res_pos_chain'].tolist()

    if len(df_CA_sub) == 0:
        # print("skip")
        continue

    if len(df_CA_sub) != len(df_CB_sub):
        continue

    coord_CA_sub = torch.tensor(df_CA_sub[['x', 'y', 'z']].to_numpy(), device=device)
    coord_CB_sub = torch.tensor(df_CB_sub[['x', 'y', 'z']].to_numpy(), device=device)

    sub_target_CA = df_CA_sub['target'].to_numpy()
    sub_target_CB = df_CB_sub['target'].to_numpy()
    sub_target_CA = torch.tensor(sub_target_CA).view(-1, 1)

    hidden_cross = cross_concatenate(hidden_CA)
    target_cross = cross_concatenate(sub_target_CA)

    ca_cb = cross_distance(coord_CA_sub, coord_CB_sub)
    cb_ca = cross_distance(coord_CB_sub, coord_CA_sub)
    cb_cb = cross_distance(coord_CB_sub, coord_CB_sub)
    ca_ca = cross_distance(coord_CA_sub, coord_CA_sub)

    pair_feat = torch.stack((ca_cb, cb_ca, cb_cb, ca_ca), dim=-1)

    # NB: Target is about pairs of residues!
    target_cross = (target_cross == 2)
    boolTarget = (target_cross[:, :, 0] * target_cross[:, :, 1]).float()

    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=10,
                                         linkage='average')
    y = clustering.fit_predict(coord_CA_sub.cpu().numpy())

    clusters = set(y)

    #print(y)
    #print(clusters)
    with torch.no_grad():
        for cluster_label in clusters:
            data_idxs = (y == cluster_label).nonzero()[0]
            #print(data_idxs)

            if len(data_idxs) > 2:

                for i in data_idxs:
                    for j in data_idxs:
                        pair = names[i]+"|"+names[j]
                        x1 = hidden_CA[i].cpu().numpy().round(4).tolist()
                        x2 = hidden_CA[j].cpu().numpy().round(4).tolist()
                        rel = pair_feat[i,j].cpu().numpy().round(4).tolist()

                        # tt = (sub_target_CA[i] == sub_target_CA[j]).float() ERROR
                        t1 = sub_target_CA[i].cpu().item()
                        t2 = sub_target_CA[j].cpu().item()

                        #df = pd.DataFrame({'pdb', 'cluster', 'x1', 'x2', 'rel', 'target1', 'target2'})
                        #print(t_pdb,cluster_label,x1, x2, rel, t1, t2)
                        #df.loc[len(df.index)] = {'pdb':t_pdb,
                        #                         'cluster': cluster_label,
                        #                         'x1': x1, 'x2': x2, 'rel':rel, 'T1': t1, 'T2':t2}
                        df.loc[len(df.index)] = [t_pdb, cluster_label, x1, x2, rel, t1, t2]

                #per ogni cluster
                #prendi gli embedding a coppie, relations tra coppie e target
                #clstr_hidden_cross = hidden_cross[data_idxs, :][:, data_idxs]
                #print(clstr_hidden_cross)


df['target'] = ((df['T1'] == 2) & (df['T2'] == 2)).astype(float)
df.to_csv("second_dataset_TEST.csv", index=False)
print(df[df['target']==1])
print(df[df['target']==0])






