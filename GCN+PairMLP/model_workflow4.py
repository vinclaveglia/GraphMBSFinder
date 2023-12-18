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
from base.wp_model import ProteinSegmenter2
import pathlib
import os
import numpy as np
import torch
from base.config import Atom
import pandas as pd
import random
from base.PDB_MBS_preprocessing import get_dataset
from base.site_classifier_model import load_site_classifier
from base.input import get_random_train_test_proteins
import warnings
import biometall

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

input_structures = train_pdb

# -------- PROCESS DATA -------------
res_df = pd.DataFrame()
n_metallo_prot = 0

#input_structures = ['1adu.pdb']

GRAPH_CONFIDENCE = 0.1

optim_graph = torch.optim.Adam(params=net_CA.parameters(), lr=0.001)

loss_fn = torch.nn.CrossEntropyLoss()


class PairMLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.enc_node = torch.nn.Linear(in_features=100, out_features=100)
        self.enc_rel = torch.nn.Linear(in_features=4, out_features=20)

        self.fc1 = torch.nn.Linear(in_features=120, out_features=200)
        self.fc2 = torch.nn.Linear(in_features=200, out_features=200)
        self.fc3 = torch.nn.Linear(in_features=200, out_features=2)

        self.ln_h_emb = torch.nn.LayerNorm(100)
        self.ln_h_rel = torch.nn.LayerNorm(20)
        self.ln_fc2 = torch.nn.LayerNorm(200)

    def forward(self,  embedding, pair_feats):

        h_emb = self.enc_node(embedding)
        h_emb = self.ln_h_emb(h_emb)

        h_rel = self.enc_rel(pair_feats)
        h_rel = self.ln_h_rel(h_rel)

        h1 = torch.cat((h_emb, h_rel), dim=-1) #+ torch.cat((embedding, pair_feats), dim=-1)

        h2 = torch.relu(self.fc1(h1))
        h2 = self.ln_fc2(h2)

        h3 = torch.relu(self.fc2(h2) + h2)

        y = self.fc3(h3)
        return y


class Pair_mlp(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(100)
        self.enc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=50))

        self.ln2 = torch.nn.LayerNorm(4)
        self.enc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=10, out_features=20))

        self.ln3 = torch.nn.LayerNorm(70)
        self.last = torch.nn.Sequential(
            torch.nn.Linear(in_features=70, out_features=70),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=70, out_features=70))

        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(in_features=70, out_features=2)


    def forward(self, embedding, pair_feats):

        embedding = self.ln1(embedding)
        h1 = self.enc1(embedding)

        pair_feats = self.ln2(pair_feats)
        h2 = self.enc2(pair_feats)

        h = torch.cat((h1, h2), dim=-1)

        h = self.ln3(h)
        last_out = self.last(h) + h
        y = self.classifier(self.relu(last_out))
        return y


pair_net = Pair_mlp()
pair_net.to(device)
optim_pairnet = torch.optim.Adam(params=pair_net.parameters(), lr=0.001)

df = pd.DataFrame(["pdb", "cluster", "concat", "relation", "target"])
dfdata_C0, dfdata_C1 = [], []

for epoch in range(200):
    tot_loss = 0.
    Loss0, Loss1 = 0., 0.

    mean_acc = 0.
    tot_acc0 = 0.
    tot_acc1 = 0.

    C0_predOK, C1_predOK = 0, 0
    C0_counter, C1_counter = 0, 0

    rnd_idxs = random.sample(range(0,len(train_pdb)), 32)
    input_structures = [train_pdb[x] for x in rnd_idxs]
    input_structures = train_pdb[:500]
    #input_structures = ['1dy0.pdb']
    #input_structures = ['4mmo.pdb']
    n_structures = len(input_structures)

    all_pred = []
    all_target = []

    n_c0, n_c1 = 0, 0

    for j, t_pdb in enumerate(input_structures):

        biometall.predict(t_pdb)

        optim_pairnet.zero_grad()
        optim_graph.zero_grad()
        print(f"========>{j}/{len(input_structures)} - {t_pdb}")

        input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=train_sites)
        input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=train_sites)

        if len(input_graph_CA) == 0 or len(input_graph_CB) == 0:
            continue

        input_graph_CA = input_graph_CA[t_pdb]
        input_graph_CB = input_graph_CB[t_pdb]

        # PREDICO I RESIDUI DI CLASSE 2
        #with torch.no_grad():
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
            #pred_res_CB2 = [x['id'] for x in pred_res_CB]
            #for x in pred_res_CA:
            #    if x['id'] in pred_res_CB2:
            #        common_res.append(x)

        common_idxs = [x['index'] for x in common_res]
        common_res_names = [x['id'] for x in common_res]

        print("common_res_names", common_res_names)

        sys.exit()

        df_CA_sub = input_graph_CA.df[input_graph_CA.df['res_pos_chain'].isin(common_res_names)]
        df_CB_sub = input_graph_CB.df[input_graph_CB.df['res_pos_chain'].isin(common_res_names)]

        if len(df_CA_sub) == 0:
            #print("skip")
            continue

        if len(df_CA_sub) != len(df_CB_sub):
            continue

        coord_CA_sub = torch.tensor(df_CA_sub[['x', 'y', 'z']].to_numpy(), device=device)
        coord_CB_sub = torch.tensor(df_CB_sub[['x', 'y', 'z']].to_numpy(), device=device)

        sub_target_CA = df_CA_sub['target'].to_numpy()
        sub_target_CB = df_CB_sub['target'].to_numpy()

        # CROSSCONTATENATION OF ALL PREDICTED RESID
        #print(coord_CA_sub.size(),  coord_CB_sub.size(),  hidden_CA.size())

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

            xdist = ((X1[None, :, :] - X2[:, None, :])**2).sum(-1).sqrt()

            return xdist

        #x_a = hidden_CA[None, :, :].expand(len(hidden_CA), -1, -1)
        #x_b = hidden_CA[:, None, :].expand(-1, len(hidden_CA), -1)
        #x_c = torch.cat((x_a, x_b), dim=2)
        sub_target_CA = torch.tensor(sub_target_CA).view(-1, 1)

        hidden_cross = cross_concatenate(hidden_CA)
        target_cross = cross_concatenate(sub_target_CA)

        ca_cb = cross_distance(coord_CA_sub, coord_CB_sub)
        cb_ca = cross_distance(coord_CB_sub, coord_CA_sub)
        cb_cb = cross_distance(coord_CB_sub, coord_CB_sub)
        ca_ca = cross_distance(coord_CA_sub, coord_CA_sub)

        pair_feat = torch.stack((ca_cb, cb_ca, cb_cb, ca_ca), dim=-1)
        pair_feat.to(device)

        # NB: Target is about pairs of residues!
        target_cross = (target_cross==2)
        boolTarget = (target_cross[:, :, 0] * target_cross[:, :, 1]).float()

        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=10,
                                             linkage='average')
        y = clustering.fit_predict(coord_CA_sub.cpu().numpy())

        clusters = set(y)
        #print("Clusters: ", clusters)
        #print("hidden_cross.size() - ", hidden_cross.size())

        for cluster_label in clusters:
            data_idxs = (y == cluster_label).nonzero()[0]

            if len(data_idxs) > 2:
                clstr_hidden_cross = hidden_cross[data_idxs,:][:,data_idxs]
                clstr_hidden_cross = clstr_hidden_cross.float()

                clstr_pair_feats = pair_feat[data_idxs,:][:,data_idxs]
                clstr_pair_feats = clstr_pair_feats.float()

                clstr_t = boolTarget[data_idxs, :][:, data_idxs]

                C0_idxs, C1_idxs = (clstr_t == 0).nonzero(), (clstr_t == 1).nonzero()
                C0_not_same = (C0_idxs[:,0] != C0_idxs[:,1]).nonzero().view(-1)
                C1_not_same = (C1_idxs[:, 0] != C1_idxs[:, 1]).nonzero().view(-1)
                C0_idxs = C0_idxs[C0_not_same]
                C1_idxs = C1_idxs[C1_not_same]

                C0_hdn_cross = clstr_hidden_cross[C0_idxs[:,0], C0_idxs[:,1]]
                C1_hdn_cross = clstr_hidden_cross[C1_idxs[:,0], C1_idxs[:,1]]

                C0_pair_feats = clstr_pair_feats[C0_idxs[:,0], C0_idxs[:,1]]
                C1_pair_feats = clstr_pair_feats[C1_idxs[:,0], C1_idxs[:,1]]

                #print(C0_hdn_cross.size(), C1_hdn_cross.size())
                #print(C0_pair_feats.size(), C1_pair_feats.size())

                #pred = pair_net(C0_hdn_cross.float(), C0_pair_feats.float())
                #print("------------------------------------------------")
                #print(clstr_hidden_cross.size(), clstr_pair_feats.size())

                pred = pair_net(clstr_hidden_cross, clstr_pair_feats)

                # print("Pred size: ", pred.size())

                if len(C0_idxs) > 0:

                    pred_C0 = pred[C0_idxs[:,0], C0_idxs[:,1]]
                    targetC0 = torch.zeros_like(pred_C0)
                    targetC0[:,0]=1.
                    Loss0 += ((pred_C0 - targetC0)**2).sum()
                    n_c0 +=1
                    C0_predOK += (pred_C0.argmax(dim=1) == 0).float().sum()
                    C0_counter += len(C0_idxs)

                    dfdata_C0.append((t_pdb, cluster_label, C0_hdn_cross, C0_pair_feats))

                if len(C1_idxs) > 0:
                    pred_C1 = pred[C1_idxs[:, 0], C1_idxs[:, 1]]
                    targetC1 = torch.zeros_like(pred_C1)
                    targetC1[:, 1] = 1.
                    Loss1 += ((pred_C1 -targetC1)  ** 2).sum()
                    n_c1 += 1
                    C1_predOK += (pred_C1.argmax(dim=1) == 1).float().sum()
                    C1_counter += len(C1_idxs)
                    dfdata_C1.append((t_pdb, cluster_label, C1_hdn_cross, C1_pair_feats))
                    #acc1 = (pred_C1 >= 0.5).float().mean()
                #print(acc0, acc1)

            else:
                #print(f"cluster label {cluster_label} ( <= 2)")
                continue

    Loss0 /= n_c0
    Loss1 /= n_c1
    Loss = Loss0 + Loss1
    Loss.backward()

    #for par in net_CA.parameters():
        #print(par)
    #print(net_CA.conv1.lin_l.weight.grad.mean())
    #print(net_CA.conv1.lin_r.weight.grad.mean())
    if epoch > 10: # se è pari
        optim_pairnet.step()
    optim_graph.step()

    with torch.no_grad():

        Acc0 = C0_predOK / C0_counter
        Acc1 = C1_predOK / C1_counter
        print(epoch, "Loss ", Loss.data.item(), Loss0.data.item(), Loss1.data.item(), Acc0.item(), Acc1.item())

    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = None
    R = None
    for pdb, cluster, concat, relation in dfdata_C0:
        #print(pdb, cluster, concat.size(), relation.size())
        if R is None:
            R = relation
            X = concat
        else:
            R = torch.cat((R, relation))
            X = torch.cat((X, concat))

    print(R.size())
    print(X.size())

    print(R.mean(dim=0))
    print(R.min(dim=0)[0])
    print(R.max(dim=0)[0])


    print("---------------")
    X1 = None
    R1 = None
    for pdb, cluster, concat, relation in dfdata_C1:
        #print(pdb, cluster, concat.size(), relation.size())
        if R1 is None:
            R1 = relation
            X1 = concat
        else:
            R1 = torch.cat((R, relation))
            X1 = torch.cat((X, concat))

    print(R1.size())
    print(X1.size())

    print(R1.mean(dim=0))
    print(R1.min(dim=0)[0])
    print(R1.max(dim=0)[0])

    XX = torch.cat((X, X1), dim=0)
    RR = torch.cat((R, R1), dim=0)

    XX = torch.cat((XX, RR), dim=1)
    XX_emb = TSNE(n_components=2).fit_transform(RR)
    print(XX_emb)
    print(XX_emb.shape)
    XX_emb0 = XX_emb[:len(X),:]
    XX_emb1 = XX_emb[len(X):, :]

    plt.plot(XX_emb0,'.')
    plt.plot(XX_emb1,'.')
    plt.savefig('tsne')

    sys.exit()
    """


"""
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
                        edge_weight= adj)

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
"""