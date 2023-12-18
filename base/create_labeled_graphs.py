import os
from base import config, input
import urllib
import sys
import numpy as np
import torch
from base.pdb_loader import load_pdb, AA_LIST
import pathlib

"""
Di base il dataset è una lista o una directory di siti.
Step-1 A partire da questi poi si scaricano i pdb (se non sono già stati scaricati)
Step-2 Per ogni PDB scaricato, si creano i csv contenenti le labels dei residui del pdb
"""

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

def preproces_protein(prot_file, filter_CHED = False):

    df_pdb = load_pdb(prot_file)

    #print(df_pdb)

    df_pdb['res_pos_chain'] = df_pdb['residue_name']+"_"+df_pdb['residue_seq_num']+"_"+df_pdb['chain_id']
    df_pdb['residue_seq_num'] = df_pdb['residue_seq_num'].astype(int)

    Phi, Psi = get_torsion_angles(df_pdb)

    # FILTRO SUI C-ALPHA
    df_pdb = df_pdb[df_pdb['atom_name'] == 'CA']
    df_pdb['Phi'] = Phi
    df_pdb['Psi'] = Psi

    # FILTRO SUGLI AMINOACIDI CHED
    if filter_CHED:
        df_pdb = df_pdb[df_pdb['residue_name'].isin(['CYS', 'HIS', 'ASP', 'GLU'])]

    df_pdb = df_pdb.assign(target=[0] * len(df_pdb))

    return df_pdb

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

def get_data(site_name, prot_name):
    print(site_name)
    print(f"site {site_name} exists?", config.SITES_PATH.joinpath(site_name).exists())
    print(f"prot {prot_name} exists?", config.PROTS_PATH.joinpath(prot_name).exists())

    if not config.PROTS_PATH.joinpath(prot_name).exists():
        print("downloading...")
        res = download_pdb(prot_name, config.PROTS_PATH)
        #print(res)

def label_protein(prot, associated_sites):
    # print(prot)

    df_CA = preproces_protein(config.PROTS_PATH.joinpath(prot))

    # print(df_CA)
    # print(df_CA['chain_id'].unique())
    #associated_sites = [x for x in sites if prot[:4] in x]
    # print(prot, associated_sites)
    involved_chains = []

    for site in associated_sites:
        dfsite = load_pdb(config.SITES_PATH.joinpath(site))

        # seleziona la catena in cui è localizzato il sito
        involved_chains += dfsite['chain_id'].unique().tolist()

        br2_res_seq_num, br1_res_seq_num, chain_, br2_RPC, br1_RPC = get_site_labels(dfsite)

        idxxx2 = df_CA[df_CA['res_pos_chain'].isin(br2_RPC)].index
        idxxx1 = df_CA[df_CA['res_pos_chain'].isin(br1_RPC)].index

        df_CA.loc[idxxx2, 'target'] = 2
        df_CA.loc[idxxx1, 'target'] = 1

    df_CA_filtered_chains = df_CA[df_CA['chain_id'].isin(involved_chains)]

    return df_CA_filtered_chains

def preprocess(site_list):

    ok = []
    wrong = []
    existing = []

    sites = [x+".site.pdb" for x in site_list]
    proteins = []
    # GET DATA
    for site in sites:
        prot = site.split('.')[0].split("_")[0]+".pdb"
        if prot not in proteins:
            proteins.append(prot)
        get_data(site, prot)

    print("/// downloading completed ///")

    pdb_path = pathlib.Path(config.PROTS_PATH)
    # LABEL PROTEINS
    for j, prot in enumerate(proteins):
        print(f'{j+1}/{len(proteins)}')

        if not pdb_path.joinpath(f'{prot}.dist').exists():
            #print(prot)
            associated_sites = [x for x in sites if prot[:4] in x]
            try:
                #df_CA = preproces_protein(config.PROTS_PATH.joinpath(prot))
                df_CAL = label_protein(prot, associated_sites)

                features = get_aminoacid_features(df_CAL['residue_name'].tolist(),
                                                  df_CAL['Phi'].tolist(),
                                                  df_CAL['Psi'].tolist())

            except Exception as err:
                wrong.append(prot+'\n')
                print("Problem with: ", prot)
                print(str(err), file=sys.stderr)
                continue

            coordinates_CA = torch.from_numpy(df_CAL[['x', 'y', 'z']].to_numpy())

            real_distances_CA = torch.sqrt(
                ((coordinates_CA[:, None, :] - coordinates_CA[None, :, :]) ** 2).sum(-1))


            np.savetxt(os.path.join(config.PROTS_PATH, f'{prot}.dist'), real_distances_CA, fmt='%.2f')
            np.savetxt(os.path.join(config.PROTS_PATH, f'{prot}.feat'), features, fmt='%.2f')
            df_CAL.to_csv(os.path.join(config.PROTS_PATH, f'{prot}.labels'), index=False)
            ok.append(prot+'\n')
            #print(df_CAL)
            #print(df_CAL.columns)
            #print(np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.feat')))
            #print(np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.feat')).shape)
        else:
            print(prot, 'already processed')
            existing.append(prot+'\n')


    with open('existings.txt', 'w') as fp:
        fp.writelines(existing)
    with open('ok.txt', 'w') as fp:
        fp.writelines(ok)
    with open('wrong.txt', 'w') as fp:
        fp.writelines(wrong)
    print("/// protein labeling completed ///")


if __name__ == '__main__':
    print("start")

    #    site_list = ['2yvr_3', '1ef0_2']
    trainsites, testsites = input.get_train_test()

    preprocess(trainsites+testsites)

