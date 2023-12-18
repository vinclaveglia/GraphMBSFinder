
"""
- processa i pdb PER SINGOLE CATENE
- e per ogni pdb i siti associati ... e li salviamo con csv .. no ? Ci conviene mi sa... senza intasarci la ram


- rivedere la loss ... ___

-

"""
from base import config
import os
import urllib
from base.pdb_loader import load_pdb, AA_LIST
import pandas as pd
import numpy as np
import sys
import torch
from torch_geometric.data import Data
from base.config import Atom

#sistemare un attimo sto coso... bug sui CB

def preproces_protein(prot_file, atom, filter_CHED = False):
    df_pdb = load_pdb(prot_file)

    df_pdb['res_pos_chain'] = df_pdb['residue_name']+"_"+df_pdb['residue_seq_num']+"_"+df_pdb['chain_id']
    df_pdb['residue_seq_num'] = df_pdb['residue_seq_num'].astype(int)

    if atom == Atom.CB:
        # FILTRO SUI C-ALPHA
        df_pdb = df_pdb[df_pdb['atom_name'] == atom]
        df_pdb['Phi'] = 0*len(df_pdb)
        df_pdb['Psi'] = 0*len(df_pdb)

    elif atom == Atom.CA:
        # CALCOLO I PHI E PSI
        Phi, Psi = get_torsion_angles(df_pdb)
        # FILTRO SUI C-ALPHA
        df_pdb = df_pdb[df_pdb['atom_name'] == atom]
        df_pdb['Phi'] = Phi
        df_pdb['Psi'] = Psi

    else:
        print("Error on atom type")


    # FILTRO SUGLI AMINOACIDI CHED
    if filter_CHED:
        df_pdb = df_pdb[df_pdb['residue_name'].isin(['CYS', 'HIS', 'ASP', 'GLU'])]

    df_pdb = df_pdb.assign(target=[0] * len(df_pdb))

    return df_pdb

def calc_dihedral(u1, u2, u3, u4):
    """ Calculate dihedral angle method. From bioPython.PDB
    (adapted to np.array)
    Calculate the dihedral angle between 4 vectors
    representing 4 connected points. The angle is in
    [-pi, pi].
    """

    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1, keepdims=True)**0.5

    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1, keepdims=True)**0.5

    #porm = np.sign((v1 * a3).sum(-1))

    rad = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)

    #if not porm == 0:
    #    rad = rad * porm

    return rad

def get_torsion_angles(df_pdb):
    # tutti i C (non CA)

    C = df_pdb[df_pdb['atom_name'] == 'C']
    N = df_pdb[df_pdb['atom_name'] == 'N']
    CA = df_pdb[df_pdb['atom_name'] == 'CA']

    # Phi
    C_i_1 = C[['x', 'y', 'z']].to_numpy()[:-1]
    N_i = N[['x', 'y', 'z']].to_numpy()[1:]
    CA_i = CA[['x', 'y', 'z']].to_numpy()[1:]
    C_i = C[['x', 'y', 'z']].to_numpy()[1:]

    #print('C_i_1', C_i_1.shape,  'N_i', N_i.shape, 'CA_i', CA_i.shape,  'C_i', C_i.shape)

    Phi = calc_dihedral(C_i_1, N_i, CA_i, C_i)
    Phi = np.insert(Phi, 0, 0.)

    # Psi
    N_i = N[['x', 'y', 'z']].to_numpy()[:-1]
    CA_i = CA[['x', 'y', 'z']].to_numpy()[:-1]
    C_i = C[['x', 'y', 'z']].to_numpy()[:-1]
    N_i_plus_1 = N[['x', 'y', 'z']].to_numpy()[1:]

    Psi = calc_dihedral(N_i, CA_i, C_i, N_i_plus_1)
    Psi = np.append(Psi, 0.)

    #print(Phi[:5])
    #print(Psi)

    return Phi, Psi

def get_site_labels(df_site, filter_CHED = False):

    df_site['res_pos_chain'] = df_site['residue_name'] \
                               + "_" + df_site['residue_seq_num'] \
                               + "_" + df_site['chain_id']

    if filter_CHED:
        df_site = df_site[df_site['residue_name'].isin(['CYS', 'HIS', 'ASP', 'GLU'])]

    # RESIDUI CORDINANTI
    df_site_br2 = df_site[df_site['temp_factor'] == 40]

    res_seq_num2 = df_site_br2['residue_seq_num'].astype(int).tolist()
    res_seq_num2_RPC = df_site_br2['res_pos_chain'].unique().tolist()

    df_site_br1 = df_site[df_site['temp_factor'] == 20]

    res_seq_num1 = df_site_br1['residue_seq_num'].astype(int).tolist()
    res_seq_num1_RPC = df_site_br1['res_pos_chain'].unique().tolist()

    chains = df_site['chain_id'].unique().tolist()

    # fix
    res_seq_num1 = [x for x in res_seq_num1 if x not in res_seq_num2]
    res_seq_num1 = list(set(res_seq_num1))

    res_seq_num2 = list(set(res_seq_num2))

    res_seq_num2.sort()
    res_seq_num1.sort()

    #print(res_seq_num2)
    #print(res_seq_num2_RPC)

    #print(res_seq_num1)
    #print(res_seq_num1_RPC)

    return res_seq_num2, res_seq_num1, chains, res_seq_num2_RPC, res_seq_num1_RPC

def get_aminoacid_features_short(aminoacids, phi, psi):

    features = np.zeros((len(aminoacids), 7))

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

       features[j, 5] = phi[j]
       features[j, 6] = psi[j]

    features = torch.from_numpy(features).float()

    return features

def get_aminoacid_features(aminoacids, phi, psi):

    features = np.zeros((len(aminoacids), 22))

    aa_dict = {}
    for i, aa in enumerate(AA_LIST):
        aa_dict[aa] = i

    #print(aa_dict)

    for j, aa_name in enumerate(aminoacids):
        # idx = configs.AMINOACIDS[aa]
        # features[j, idx] = 1.

        oh_idx = aa_dict[aa_name]

        features[j, oh_idx] = 1.

        features[j, 20] = phi[j]
        features[j, 21] = psi[j]

    features = torch.from_numpy(features).float()

    return features

def create_graph_object(features, adj_matrix, node_target=None, df= None, name = "prova"):

    edge_index = adj_matrix.nonzero().t().contiguous().long()

    edge_weight = adj_matrix[edge_index.tolist()] # is a 1D array ???

    x = features

    sub_df = df[['atom_name', 'residue_name', 'res_pos_chain', 'x', 'y', 'z', 'target']].reset_index(drop=True)
    #sub_df = df

    if node_target != None:
        data = Data(x=x, df=sub_df, edge_index=edge_index, edge_weight=edge_weight, y=node_target)
    else:
        data = Data(x=x, df=sub_df, edge_index=edge_index, edge_weight=edge_weight)

    data["name"] = name
    data['residues'] = df['res_pos_chain'].tolist()

    return data

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    #pdbfn = pdbcode + ".pdb"
    pdbfn = pdbcode
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def get_data(prot_name):
    #print(site_name)
    #print(f"site {site_name} exists?", config.SITES_PATH.joinpath(site_name).exists())
    #print(f"prot {prot_name} exists?", config.PROTS_PATH.joinpath(prot_name).exists())

    if not config.PROTS_PATH.joinpath(prot_name).exists():
        print("downloading...")
        res = download_pdb(prot_name, config.PROTS_PATH)
        #print(res)

#if __file__ == '__main__':

###############################################################


def get_dataset(proteins,
                atom,
                sites=[],
                dataframe_only=False,
                path_to_structures=config.PROTS_PATH, # temporary
                path_to_sites=config.SITES_PATH): # temporary

    #for _p in proteins:
    #    get_data(_p)

    # LABEL PROTEINS
    DFS = {}
    for prot in proteins:
        #print(prot)
        try:
            df_CA = preproces_protein(path_to_structures.joinpath(prot), atom)
            #df_CA = preproces_protein(config.PROTS_PATH.joinpath(prot))
        except:
            print(">> Problem with: ", prot)
            continue
        #print("---")
        #print(df_CA)
        #print(df_CA['chain_id'].unique())
        associated_sites = [f'{x}.site.pdb' for x in sites if prot[:4] in x]

        involved_chains = []

        # Load target values
        for site in associated_sites:

            #dfsite = load_pdb(config.SITES_PATH.joinpath(site))
            dfsite = load_pdb(path_to_sites.joinpath(site))

            # seleziona la catena in cui è localizzato il sito
            involved_chains += dfsite['chain_id'].unique().tolist()

            br2_res_seq_num, br1_res_seq_num, chain_, br2_RPC, br1_RPC = get_site_labels(dfsite)

            idxxx2 = df_CA[df_CA['res_pos_chain'].isin(br2_RPC)].index
            idxxx1 = df_CA[df_CA['res_pos_chain'].isin(br1_RPC)].index

            df_CA.loc[idxxx2, 'target'] = 2
            df_CA.loc[idxxx1, 'target'] = 1

        if len(involved_chains) > 0:
            # significa che ci sono i siti target
            df_CA_filtered_chains = df_CA[df_CA['chain_id'].isin(involved_chains)]
        else:
            # prendiamo tutte le catene (i residui non sono etichettati)
            df_CA_filtered_chains = df_CA

        #DFS.append((prot, df_CA_filtered_chains))
        DFS[prot] = df_CA_filtered_chains

    #for j,k in DFS:
    #    print(j)
    #    print(k)

    if dataframe_only == True:
        return DFS

    # FEATURES AND ADJ MATRIX
    GRAPH_DATASET = {}

    for prot, df_CA in DFS.items():

        residues_CA = df_CA['residue_name'].tolist()
        try:
            features = get_aminoacid_features(residues_CA, df_CA['Phi'].tolist(), df_CA['Psi'].tolist())
        except:
            print("Problem aminoacid feature for", prot)
            continue


        # prende la colonna target del df
        Y_A = torch.from_numpy(df_CA['target'].to_numpy())

        # la trasforma in onehot
        target_CA = torch.zeros(len(Y_A), 3)
        for i,v in enumerate(Y_A):
            target_CA[i, v] = 1.

        coordinates_CA = torch.from_numpy(df_CA[['x', 'y', 'z']].to_numpy())

        real_distances_CA = torch.sqrt(
            ((coordinates_CA[:, None, :] - coordinates_CA[None, :, :]) ** 2).sum(-1))

        distances_CA = torch.exp(-real_distances_CA/5).float()

        mm = real_distances_CA <= 10 # 8 # ???

        distances_CA *= mm


        # todo passandogli distances_CA sto coso funziona a culo!!!
        graph = create_graph_object(features, adj_matrix=distances_CA, node_target=target_CA, name=prot, df=df_CA)

        #g = torch_geometric.utils.to_networkx(graph, to_undirected=True)

        #pos = nx.spring_layout(g, seed=42)
        #nx.draw_networkx_edges(g, pos, width=0.4, alpha=0.3)
        #nx.draw(g, )

        #plt.show()

        GRAPH_DATASET[prot] = graph

    return GRAPH_DATASET

#get_dataset()




















