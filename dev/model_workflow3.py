"""

- model CA (TRAINED)

- model CB (TRAINED)

TO DO
- A partire dai residui predetti come binding-role-2, si costruisce un nuovo grafo prendendo sia i CA che i CB.
Una GCN viene addestrata a predirre quali sono i residui che compongono il sito.
In questo caso i grafi sono molto più piccoli... dovrebbe/potrebbe funzionare


"""
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
import sys
import pandas as pd
import random
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


def predict_site(input_structure, model, threshold=0.7):

    _, pred = model(input_structure)

    pred = torch.softmax(pred, dim=1)

    pred2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    #adj = torch_geometric.utils.to_dense_adj(input_structure.edge_index)[0]

    # predicted binding residues info
    pred_res0 = [{'id':input_structure.residues[j_idx],
                  #'n_vicini': int(n_vicini[j_idx])-1,
                  'confidence':np.round(pred[j_idx, 2].item(), 4),
                  'index':j_idx.item()}
                 for j_idx in pred2_idxs ]

    pred_res0 = [d for d in pred_res0 if d['confidence'] >= threshold]

    return pred_res0


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
#test_pdb = test_pdb[:50]
#train_pdb = train_pdb[:50]
train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]


input_structures = train_pdb


# -------- PROCESS DATA -------------
res_df = pd.DataFrame()
n_metallo_prot = 0


#input_structures = ['1adu.pdb']

GRAPH_CONFIDENCE = 0.1

secondNet = SecondGCN(num_node_features=22, num_classes=2)

optim = torch.optim.Adam(params=secondNet.parameters(), lr=0.01)

loss_fn = torch.nn.CrossEntropyLoss()




for epoch in range(50):
    tot_loss = 0.
    mean_acc = 0.
    tot_acc0 = 0.
    tot_acc1 = 0.

    rnd_idxs = random.sample(range(0,len(train_pdb)), 5)
    input_structures = [train_pdb[x] for x in rnd_idxs]
    #input_structures = train_pdb[:5]

    n_structures = len(input_structures)

    all_pred = []
    all_target = []

    for j, t_pdb in enumerate(input_structures):

        #print("========>",t_pdb)

        #input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=[], path_to_structures=structures_path)
        #input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=[], path_to_structures=structures_path)

        input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=train_sites)
        input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=train_sites)


        if len(input_graph_CA) == 0 or len(input_graph_CB) == 0:
            continue

        input_graph_CA = input_graph_CA[t_pdb]
        input_graph_CB = input_graph_CB[t_pdb]


        # PREDICO I RESIDUI DI CLASSE 2
        with torch.no_grad():
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

        common_idxs = [x['index'] for x in common_res]

        sub_target = input_graph_CA.y.argmax(dim=1).view(-1)[common_idxs]

        df_CA_sub = input_graph_CA.df.iloc[common_idxs]

        rresid = df_CA_sub['res_pos_chain'].tolist()

        df_CB_sub = input_graph_CB.df[input_graph_CB.df['res_pos_chain'].isin(rresid)]

        #print(len(df_CA_sub), len(df_CB_sub))

        # GETTING FEATURES
        df_CB_sub_idxs = df_CB_sub.index.values
        feats_CA = input_graph_CA.x[common_idxs]
        feats_CB = input_graph_CB.x[df_CB_sub_idxs]

        features = torch.cat((feats_CA[:,:20], feats_CB[:,:20]), dim=0)
        AB = torch.zeros(2 * len(df_CA_sub), 2)
        AB[0:len(df_CA_sub), 0] = 1.
        AB[len(df_CA_sub):, 1] = 1.
        features = torch.cat((features, AB), dim=1)

        # GET DISTANCE MATRIX
        adj = CACB_distance_matrix(input_graph_CA, input_graph_CB, rresid)
        mask = (adj <= 10).float()

        # GET TARGET
        sub_target_CB = input_graph_CB.y.argmax(dim=1).view(-1)[df_CB_sub_idxs]

        TTarget = torch.cat((sub_target, sub_target_CB), dim=0)

        TTarget[TTarget==1] = 0
        TTarget[TTarget == 2] = 1

        TTarget_oh = torch.zeros(len(TTarget), 2)
        for j, v in enumerate(TTarget):
            TTarget_oh[j, v] = 1.

        # todo: NOTA BENE
        # elevarlo alla -15 aggiunge troppo rumore...
        # da importanza anche ai nodi lontani più di 10A, che poi sarebbe la soglia consentita
        # rivedere questo valore anche nella vecchia versione/impostazione
        # forse è uno dei problemi dei grafi di input!!!
        adj = torch.exp(-adj/5)

        thegraph = Data(x=features,
                        edge_index=adj.nonzero().t().contiguous().long(),
                        y=TTarget_oh,
                        edge_weight= adj) #

        ppred = secondNet(thegraph)
        #print(ppred)
        C0_idxs = (thegraph.y.argmax(dim=1) == 0).nonzero().view(-1)
        C1_idxs = (thegraph.y.argmax(dim=1) == 1).nonzero().view(-1)

        if len(C0_idxs) == 0:
            continue
        if len(C1_idxs) == 0:
            continue
        #print(C0_idxs)
        #print(C1_idxs)

        loss0 = ((ppred[C0_idxs] - thegraph.y[C0_idxs]) ** 2).mean()
        loss1 = ((ppred[C1_idxs] - thegraph.y[C1_idxs]) ** 2).mean()

        #loss0 = loss_fn(ppred[C0_idxs], thegraph.y[C0_idxs].argmax(dim=1))
        #loss1 = loss_fn(ppred[C1_idxs], thegraph.y[C1_idxs].argmax(dim=1))

        loss = loss0 + loss1

        with torch.no_grad():
            acc0 = (ppred[C0_idxs].argmax(dim=1).view(-1) == 0).view(-1).float().mean()
            acc1 = (ppred[C1_idxs].argmax(dim=1).view(-1) == 1).view(-1).float().mean()

            #print(acc0, acc1)
            #mean_acc+=acc

            tot_acc0 += acc0
            tot_acc1 += acc1

        tot_loss += loss

    tot_acc0 = tot_acc0/n_structures
    tot_acc1 = tot_acc1 / n_structures
    print(f"epoch {epoch} tot loss ", tot_loss.data, "acc", tot_acc0, tot_acc1)

    tot_loss.backward()

    optim.step()

print('fine')
