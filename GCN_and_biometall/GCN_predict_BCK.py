
import sys
sys.path.insert(0, '..')
from base.wp_model import ProteinSegmenter2
from base.pdb_loader import load_pdb
from base import config
import torch_geometric
import numpy as np
import torch
from base.config import Atom
import pandas as pd
from base.PDB_MBS_preprocessing import get_dataset
from base.input import get_random_train_test_proteins
import warnings
import biometall


def predict_site(input_structure, model, threshold=0.7, hidden=False):

    with torch.no_grad():
        hidden, pred = model(input_structure) #

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


def set_vicini(pred_res_, adj_):
    #print([d['index'] for d in pred_res_])
    sub_idxs = torch.tensor( [d['index'] for d in pred_res_] )

    # count neughtbors
    sub_adj = adj_[:, sub_idxs]
    n_vicini = sub_adj.sum(dim=1).tolist()

    # da integrare con il pezzo di sopra
    for d in pred_res_:
        j_idx = d['index']
        d['n_vicini'] = int(n_vicini[j_idx]) - 1 # perchè prima ha contato pure se stesso
        sub_vicini_idxs = sub_adj[j_idx].nonzero().view(-1)
        d['vicini_idxs'] = sub_idxs[sub_vicini_idxs]


warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

GRAPH_CONFIDENCE = 0.1

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

# -------- LOAD DATA ----------------
# Load trainin data
train_pdb, test_pdb, db_sites = get_random_train_test_proteins()
#test_pdb = test_pdb[:50]
#train_pdb = train_pdb[:50]
train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

#input_structures = train_pdb[500:510]
input_structures = test_pdb[100:150]
input_structures = test_pdb
#input_structures = ['5wjq.pdb']
input_structures = ['4zxn.pdb']

res_df = pd.DataFrame(columns=['pdb', 'target'])

#EVALUATING THE RESULT
for j, t_pdb in enumerate(input_structures):
    print(f"======== >{j}/{len(input_structures)} - {t_pdb}")

    input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=db_sites) #
    input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=db_sites) #dddd

    if len(input_graph_CA) == 0 or len(input_graph_CB) == 0:
        continue

    df_all = load_pdb(config.PROTS_PATH.joinpath(t_pdb)) ###

    input_graph_CA = input_graph_CA[t_pdb]
    input_graph_CB = input_graph_CB[t_pdb]

    df_CA = input_graph_CA.df
    df_CB = input_graph_CB.df

    Tca = df_CA[df_CA['target']==2]['res_pos_chain'].tolist()
    Tca = {x: False for x in Tca}
    #Tcb = df_CB[df_CB['target'] == 2]['res_pos_chain'].tolist()
    #print("TARGET")
    #print(list(Tca.keys()))
    #print(Tcb)

    print("LEN: ", len(df_CA))
    if len(df_CA) > 500:
        continue

    # PREDICO I RESIDUI DI CLASSE 2
    input_graph_CA.to(device)
    input_graph_CB.to(device)

    pred_res_CA, hidden_CA = predict_site(input_graph_CA, net_CA, threshold=GRAPH_CONFIDENCE, hidden=True)
    pred_res_CB, hidden_CB = predict_site(input_graph_CB, net_CB, threshold=GRAPH_CONFIDENCE, hidden=True)

    if len(pred_res_CA) < 3:
        print(pred_res_CA, "NOT ENOUGH RESIDUES")
        continue

    pred_res_union = [x['id'] for x in pred_res_CA] + [x['id'] for x in pred_res_CB]

    #common_res = []
    #print(pred_res_CA)

    adj_CA = torch_geometric.utils.to_dense_adj(input_graph_CA.edge_index)[0]
    # non deve dipendere da come è stata creata adj_CA!!! andrebbero riviste le distanze

    set_vicini(pred_res_CA, adj_CA)

    res_non_isolati = [x for x in pred_res_CA if x['n_vicini'] >= 2]
    print("NON ISOLATI", [x['id'] for x in res_non_isolati])
    #res_non_isolati_idx = [x['index'] for x in res_non_isolati]
    #res_non_isolati_names = [x['id'] for x in res_non_isolati]
    res_non_isolati_names = pred_res_union
    #print("NON ISOLATI", res_non_isolati_names)
    print("UNION", res_non_isolati_names)
    df_CA_pred = df_CA[df_CA['res_pos_chain'].isin(res_non_isolati_names)]
    df_CB_pred = df_CB[df_CB['res_pos_chain'].isin(res_non_isolati_names)]

    #print(df_CA_pred)

    coord_CA = df_CA_pred[['x', 'y', 'z']].to_numpy()

    #forse il clustering meglio non metterlo per ora ...
    print("- - - " * 20)
    pred = biometall.predict(df_CA_pred, df_CB_pred, df_all, res_non_isolati_names)
    #print("///////////////")

    site_lenghts = sorted([int(x) for x in list(pred.keys())], reverse=True)

    for l1 in site_lenghts:
        #print(".........................")
        for dict1 in pred[l1]:
            site1 = dict1['residues']
            #print("::::", site1)
            for l2 in site_lenghts:
                if l2 < l1:
                    for dict2 in pred[l2]:
                        site2 = dict2['residues']
                        #if all([True if x in site1 else False for x in site2 ]):
                        if all([x in site1 for x in site2]):
                            #allora scarta
                            dict2['included'] = True
                            #print(site2, "INCLUDED")


    for l1 in site_lenghts:
        #print(".........................")
        for dict1 in pred[l1]:
            site1 = dict1['residues']
            #print("::::", site1)
            for l2 in site_lenghts:
                if l2 < l1:
                    for dict2 in pred[l2]:
                        site2 = dict2['residues']
                        #if all([True if x in site1 else False for x in site2 ]):
                        #if overlap(site1, site2):
                        if (sum([1 if x in site1 else 0 for x in site2]) >= (len(site2)-1)):
                            #allora scarta
                            dict2['overlap'] = True
                            #print(site2, "INCLUDED")

    RES = []
    for l in site_lenghts:
        for d in pred[l]:
            if 'included' not in d.keys():
                if 'overlap' not in d.keys():
                    print("()",d)
                    RES.append(d)


    for h in RES:
        p_res = h['residues']

        # se ne becca almeno 3
        if sum([1 if x in Tca.keys() else 0 for x in p_res]) >= 3:
            # segna come hit quelli che ho preso
            for x in p_res:
                if x in Tca.keys():
                    Tca[x] = True

    print(Tca)
    res_df.loc[len(res_df)] = [t_pdb, Tca]


    #come ha detto claudia


    #clustering = AgglomerativeClustering(n_clusters=None,
    #                                     distance_threshold=10,
    #                                     linkage='average')

    '''y = clustering.fit_predict(coord_CA)
    print("Y ",y)
    clusters = set(y)
    print("- - - "*20)
    for cluster_label in clusters:
        data_idxs = (y == cluster_label).nonzero()[0]
        print(f"Cluster label {cluster_label}, n. elements {len(data_idxs)}, {data_idxs}")
        if len(data_idxs) > 2:
            #questi vanno indicizzati in base al cluster
            #biometall.predict(df_CA_pred, df_CB_pred)
            df_CA_pred2 = df_CA_pred.iloc[data_idxs]
            df_CB_pred2 = df_CB_pred.iloc[data_idxs]
            print(df_CA_pred2)
            biometall.predict(df_CA_pred, df_CB_pred, df_all)
            print("---"*30)

    #servono gli indici per selezionare le coordinate dei residui predetti
    '''

#todo 1 - predizione con la GCN_CA

#todo 2 - togliere quelli a bassa confidenza

#todo 3 - togliere i residui isolati

#A) => Metterli tutti
#B) => Aggregare i metalli predetti che distano tra loro meno di 1 (o 2?) armstr., insomma quelli che si overlappano
#e concatenarne i residui associati, overlapping di almeno 2 residui.

res_df.to_csv("prova.out")