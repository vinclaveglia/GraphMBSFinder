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
from base.evaluation import evaluate
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


def rimuovi_isolati(pred_res_, adj_, min_n_vicini=1):
    # only residues with high confidence
    #pred_res1 = [d for d in pred_res_ if d['confidence'] >= threshold_]
    sub_idxs = np.array([d['index'].item() for d in pred_res_])



    # count neughtbors
    sub_adj = adj_[:, sub_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()


    # da integrare con il pezzo di sopra
    for d in pred_res_:
        j_idx = d['index']
        d['n_vicini'] = int(n_vicini[j_idx]) - 1
        sub_vicini_idxs = sub_adj[j_idx].nonzero().view(-1).numpy()
        d['vicini_idxs'] = sub_idxs[sub_vicini_idxs]



    # final
    pred_res_out = []
    bbb = []
    for res_dict in pred_res_:
        if ( res_dict['n_vicini'] > min_n_vicini ):
            pred_res_out.append(
                f"{res_dict['id']}|{res_dict['n_vicini']}|{res_dict['confidence']}")
            bbb.append(res_dict)


    return pred_res_out, bbb


def predict_site(input_structure, model, threshold=0.7):

    embeddings, pred = model(input_structure)

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



# LOAD TRAINED MODELS
# load trained model on CA graph
net_CA = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)
# load trained model on CB graph
net_CB = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = 50)

net_CA_path = '../metadata/model_0.01_50_2_5_0.pth'
net_CB_path = 'grid_search/model_0.01_50_2_5_0.CB.pth_FP0'

net_CA.load_state_dict(torch.load(net_CA_path, map_location=torch.device('cpu')))
net_CB.load_state_dict(torch.load(net_CB_path, map_location=torch.device('cpu')))


# LOAD TRAINING DATASET
train_pdb, test_pdb, db_sites = get_random_train_test_proteins()

test_pdb = test_pdb[:50]
train_pdb = train_pdb[:50]

train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

res_df = pd.DataFrame()
n_metallo_prot = 0

class FinalClassifier(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=input_dim, out_features=60)
        self.fc2 = torch.nn.Linear(in_features=60, out_features=3)
        self.ln = torch.nn.LayerNorm(input_dim)

    def forward(self, x):

        x = self.ln(x)

        x = torch.relu(self.fc1(x))

        return self.fc2(x)

finalclassifier = FinalClassifier(input_dim=100)

optim_final = torch.optim.Adam(finalclassifier.parameters(), lr=0.01)
optim_CA = torch.optim.Adam(net_CA.parameters(), lr=0.001)
optim_CB = torch.optim.Adam(net_CB.parameters(), lr=0.001)

#per adesso online learning ... un paio di epoche

for epoch in range(10):
    # Iterate over the input structures list
    print(f"epoch {epoch}")
    ll = len(train_pdb)
    tot_loss = 0.
    for j, t_pdb in enumerate(train_pdb):

        optim_final.zero_grad()
        optim_CA.zero_grad()
        optim_CB.zero_grad()

        #print(f"========> {j+1}/{ll}",t_pdb )
        input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=train_sites)
        input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=train_sites)
        if len(input_graph_CA)==0 or len(input_graph_CB)==0:
            #print("=="*50, t_pdb)
            continue
        input_graph_CA = input_graph_CA[t_pdb]
        input_graph_CB = input_graph_CB[t_pdb]
        _target0 = input_graph_CA.y.argmax(dim=1)

        # index of glycine residues
        noncb = [x for x in input_graph_CA.residues if x not in input_graph_CB.residues] # residui che non hanno il CB
        noncb_idxs = input_graph_CA.df[input_graph_CA.df['res_pos_chain'].isin(noncb)].index.to_numpy() # relativi indici in df_CA

        # PREDICO I RESIDUI DI CLASSE 2
        embeddings_CA, pred_CA = net_CA(input_graph_CA)
        embeddings_CB, pred_CB = net_CB(input_graph_CB)

        # get glycines idxs
        ca_idxs = np.arange(len(embeddings_CA))
        ca_no_gly_idxs = np.array([x for x in ca_idxs if x not in noncb_idxs])
        # remove glycines associated rows
        embeddings_CA = embeddings_CA[ca_no_gly_idxs]
        input_graph_CA.y = input_graph_CA.y[ca_no_gly_idxs]
        # now we can concatenate
        embeddings_concat = torch.cat( (embeddings_CA, embeddings_CB), dim=1 )
        #embeddings_concat = embeddings_CA + embeddings_CB

        final_pred = finalclassifier(embeddings_concat)

        _target = input_graph_CB.y.argmax(dim=1)

        _adj_CA0 = torch_geometric.utils.to_dense_adj(input_graph_CA.edge_index)[0]
        _adj_CA = _adj_CA0[ca_no_gly_idxs,:][:, ca_no_gly_idxs]
        #print(_adj_CA.size())

        # -| settare bene i lambda_C0 e lambra_group |-

        if j <10:
            with torch.no_grad():
                loss_CA, loss_dict_CA = evaluate(pred_CA, _target0, _adj_CA0, epoch=epoch, lambda_C0=2, lambda_group=5)
                loss_CB, loss_dict_CB = evaluate(pred_CB, _target, _adj_CA, epoch=epoch, lambda_C0=2, lambda_group=5)
                print("by CA graph", loss_dict_CA)
                print("by CB graph", loss_dict_CB)

        loss, loss_dict = evaluate(final_pred, _target, _adj_CA, epoch=epoch, lambda_C0=2, lambda_group=5)

        tot_loss+=loss
        if j<10:
            print("by actual", loss_dict)
            print('----------')

    print(tot_loss)


    tot_loss.backward()
    optim_final.step()
    optim_CA.step()
    optim_CB.step()


'''

    sys.exit()
    """"""
    common_res = []

    # Prendo quelli in comune a CA e CB
    if ((len(pred_res_CA) >= 2) and (len(pred_res_CB) >= 2)):
        # PRENDI QUELLI IN COMUNE A CA E CB
        pred_res_CB2 = [x['id'] for x in pred_res_CB]
        for x in pred_res_CA:
            if x['id'] in pred_res_CB2:
                common_res.append(x)

    # Rimuovo quelli isolati
    adj_CA = torch_geometric.utils.to_dense_adj(input_graph_CA.edge_index)[0]
    out, bbb = rimuovi_isolati(common_res, adj_CA)
    out2 = [x.split('|')[0] for x in out]

    #predittore gcn del mom1
    #per ogni residuo predetto... prendere i vicini e vedere se compongono un sito

    for x in bbb:
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
        net = load_site_classifier('metadata/trained_model_F0.pth')

        #hidden, net_out = net(potential_mbs.x, potential_mbs.adj.float())

        #print(f'features {features.size()}, distances_CACB {distances_CACB.size()}')
        hidden, net_out = net(features, distances_CACB)

        P_mbs = round(net_out[1].item(), 4)

        x['pred_mbs'] = P_mbs



    #come comportarsi adesso?
    #ogni aminoacido predetto con si, appartiene a un sito...
    outt = [f"{x['id']}|{x['n_vicini']}|{x['confidence']}" for x in bbb if  x['pred_mbs'] > 0.6 ]


    binding_residues = ';'.join(outt)

    if len(outt) > 2:
        res_df = res_df.append({'input_structure':t_pdb, 'n_binding_res': len(outt),
                                'predicted': binding_residues}, ignore_index=True)
        n_metallo_prot += 1
    else:
        res_df = res_df.append({'input_structure': t_pdb, 'n_binding_res': len(outt),
                                'predicted': 'NO_SITES_PREDICTED'}, ignore_index=True)

    print(f'{j}/{len(input_structures)} - {t_pdb} - # metallo prot {n_metallo_prot}')


res_df.to_csv('MasterOfMetal2_pred_08nov.ALL.txt', index=False)


'''

## toccherebbe capire ogni modulo quante ne butta fuori / scarta ...
