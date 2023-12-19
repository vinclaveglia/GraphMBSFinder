
import sys
sys.path.insert(0, '..')
#from base.wp_model import ProteinSegmenter2
#from base.pdb_loader import load_pdb
from base import config
import torch_geometric
import numpy as np
import torch
from base.config import Atom
import pandas as pd
from base.PDB_MBS_preprocessing import get_dataset
from base.input import get_random_train_test_proteins
import warnings
#import biometall
from GCN_and_biometall.gnn_biometall_workflow import *



warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# ------ LOAD TRAINED MODELS --------
net_CA_path = '../metadata/model_0.01_50_2_5_0.pth'
net_CB_path = '../grid_search/model_0.01_50_2_5_0.CB.pth_FP0'

net_CA = load_trained_gnn(net_CA_path)
net_CB = load_trained_gnn(net_CB_path)

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
#input_structures = ['4zxn.pdb']

path_to_structures = config.PROTS_PATH
path_to_sites = config.SITES_PATH

# ----------------------------------
#path_to_structures = "path to alphafold structures"
db_sites = []


res_df = pd.DataFrame(columns=['pdb', 'target'])

for j, t_pdb in enumerate(input_structures):

    print(f"======== >{j}/{len(input_structures)} - {t_pdb}")

    try:
        #input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=db_sites, path_to_structures=, path_to_sites=)  #
        input_graph_CA = get_dataset([t_pdb], atom=Atom.CA, sites=db_sites,
                                     path_to_structures=path_to_structures, path_to_sites=path_to_sites)  #
        input_graph_CB = get_dataset([t_pdb], atom=Atom.CB, sites=db_sites,
                                     path_to_structures=path_to_structures, path_to_sites=path_to_sites)  # dddd

        input_graph_CA = input_graph_CA[t_pdb]
        input_graph_CB = input_graph_CB[t_pdb]

    except:
        print("continue...")
        continue

    input_graph_CA.to(device)
    input_graph_CB.to(device)

    df_CA = input_graph_CA.df
    #df_CB = input_graph_CB.df
    Tca = df_CA[df_CA['target'] == 2]['res_pos_chain'].tolist()
    Tca = {x: False for x in Tca}

    pred = gnn_bmtl_predict(t_pdb, input_graph_CA, input_graph_CB, net_CA, net_CB, path_to_structures)
    if pred == None:
        continue

    include_and_overlap(pred)

    site_lenghts = sorted([int(x) for x in list(pred.keys())], reverse=True)
    RES = []
    for l in site_lenghts:
        for d in pred[l]:
            if 'included' not in d.keys():
                if 'overlap' not in d.keys():
                    print("()",d)
                    RES.append(d)


    """
    # Target evaluation:
    for h in RES:
        p_res = h['residues']
        # se ne becca almeno 3
        if sum([1 if x in Tca.keys() else 0 for x in p_res]) >= 3:
            # segna come hit quelli che ho preso
            for x in p_res:
                if x in Tca.keys():
                    Tca[x] = True

    print(Tca)
    # store in the dataframe
    res_df.loc[len(res_df)] = [t_pdb, Tca]
    """
#res_df.to_csv("prova.out")






# => Aggregare i metalli predetti che distano tra loro meno di 1 (o 2?) armstr.,
# insomma quelli che si overlappano
#e concatenarne i residui associati, overlapping di almeno 2 residui.

