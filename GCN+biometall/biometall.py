import sys
import torch
from base import config
from base.pdb_loader import load_pdb, AA_LIST
from base.config import Atom
from base.PDB_MBS_preprocessing import preproces_protein

device = 'cuda' if torch.cuda.is_available() else 'cpu'


CONDITIONS_CA = {'ASP': [3.907, 6.192],
              'HIS': [3.076, 7.098],
              'GLU': [3.591, 8.303],
              'CYS': [3.248, 5.451]}

CONDITIONS_CB = {'ASP': [3.658, 5.052],
              'HIS': [3.047, 6.073],
              'GLU': [4.123, 6.108],
              'CYS': [2.794, 3.829]}

CONDITIONS_alpha = {'ASP': [0.003, 1.871],
              'HIS': [0.001, 2.018],
              'GLU': [0.000, 2.305],
              'CYS': [0.006, 1.734]}



def generate_grid(min_x, max_x, min_y, max_y, min_z, max_z, step):
    "Generate spherical grid of probes"
    offset = 8
    xs = torch.arange(min_x-offset, max_x+offset+1, step)
    ys = torch.arange(min_y-offset, max_y+offset+1, step)
    zs = torch.arange(min_z-offset, max_z+offset+1, step)

    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs)

    Grid = torch.stack((grid_x, grid_y, grid_z), dim=0)

    #grid_x, grid_y, grid_z = Grid[0], Grid[1], Grid[2]

    return Grid


def get_distances(grid_2d, residues):
    dd = (grid_2d[:, None, :] - residues[None, :, :]) ** 2  # (N x k x 3)
    # somma lungo le componeneti
    dd = dd.sum(-1)  # N x K
    # matrice delle distanze
    dd = torch.sqrt(dd)  # N x K

    return dd

#def predict(prot):
def predict(df_CA, df_CB, df_all, predicted_res):
    print("BIOMETALL")
    coord_all = torch.from_numpy(df_all[['x', 'y', 'z']].to_numpy()).float().to(device)
    #df_CA = preproces_protein(config.PROTS_PATH.joinpath(prot), Atom.CA)
    #df_CB = preproces_protein(config.PROTS_PATH.joinpath(prot), Atom.CB)

    #df_CA = preproces_protein(prot, Atom.CA)
    #df_CB = preproces_protein(prot, Atom.CB)

    df_CA = df_CA[df_CA['residue_name']!='GLY']
    df_CA = df_CA[ df_CA['residue_name'].isin(['ASP', 'HIS', 'GLU', 'CYS'])]
    df_CB = df_CB[df_CB['residue_name'].isin(['ASP', 'HIS', 'GLU', 'CYS'])]

    res_names = df_CA['residue_name'].to_numpy()
    res_pos_chain = df_CA['res_pos_chain'].to_numpy()

    residues_CA = torch.from_numpy(df_CA[['x', 'y', 'z']].to_numpy()).detach().to(device)
    residues_CB = torch.from_numpy(df_CB[['x', 'y', 'z']].to_numpy()).detach().to(device)

    molecule_centroid = residues_CA.mean(dim=0)
    #print("molecule_centroid", molecule_centroid)

    max_x, max_y, max_z = residues_CA.max(dim=0)[0]
    min_x, min_y, min_z = residues_CA.min(dim=0)[0]

    grid = generate_grid(min_x, max_x, min_y, max_y, min_z, max_z, step=1).detach().to(device)
    grid_2d = grid.reshape(3, -1).T # funziona perchè c'è questo reshape

    #print(grid_2d)
    #residues = get_residues()
    prob_CA_distances = get_distances(grid_2d, residues_CA)#  size: (#probes x #residui)

    prob_CB_distances = get_distances(grid_2d, residues_CB)  # size: (#probes x #residui)

    #print(prob_CA_distances.size())
    #print(prob_CB_distances.size())

    #CA_CB_distances = get_distances(residues_CA, residues_CB) # size: (#residue_CA x #residue_CB) ???

    CA_CB = torch.sqrt( ((residues_CA - residues_CB)**2).sum(-1)) # size: #residue_CA

    #print(CA_CB.size())

    cos_alpha = ( prob_CA_distances**2 + CA_CB**2 - prob_CB_distances**2 )/ (2*prob_CA_distances*CA_CB)
    # size: (#probes x #residui)

    arccos_alpha = torch.arccos(cos_alpha)

    MASK_CA = torch.zeros_like(prob_CA_distances).bool() # (#probes x #residui)
    MASK_CB = torch.zeros_like(prob_CB_distances).bool() # (#probes x #residui)
    MASK_alpha = torch.zeros_like(cos_alpha).bool() # (#probes x #residui)

    for aa in ['ASP', 'HIS', 'GLU', 'CYS']:
        idxs_ = (res_names == aa).nonzero()[0]

        #print(CONDITIONS[aa][0], CONDITIONS[aa][1])
        MASK_CA[:, idxs_] = (prob_CA_distances[:, idxs_] >= CONDITIONS_CA[aa][0]) \
                         & (prob_CA_distances[:, idxs_] <= CONDITIONS_CA[aa][1])

        MASK_CB[:, idxs_] = (prob_CB_distances[:, idxs_] >= CONDITIONS_CB[aa][0]) \
                         & (prob_CB_distances[:, idxs_] <= CONDITIONS_CB[aa][1])

        MASK_alpha[:, idxs_] = (arccos_alpha[:, idxs_] >= CONDITIONS_alpha[aa][0]) \
                               &  (arccos_alpha[:, idxs_] <= CONDITIONS_alpha[aa][1])

    #mask = (pr_distances < 6.192) & (pr_distances > 3.907) # (è un esempio...)
    mask = MASK_CA * MASK_CB * MASK_alpha
    #print("mask size ", mask.size())

    """
    Matrice in cui ogni riga è un site_profile, le colonne corrispondono ai singoli RESIDUI coinvolti
    """
    # contiene i site profiles (ovveri i set di residui che potenzialmente formano un sito) possibili
    site_profile_RESIDUES = mask.int().unique(dim=0)# (#profili x #residui)
    # filtra sui prifili in cui sono coinvolti almeno 2 residui
    site_profile_RESIDUES = site_profile_RESIDUES[site_profile_RESIDUES.sum(dim=1) > 2]

    #print(site_profile_RESIDUES.size())

    """
    Matrice contenente le PROBES associate ai diversi site_profile,
    ogni riga è associata a un site_profile, 
    le colonne sono True in corrispondenza delle PROBES che hanno quel site_profile.
    La prima riga è associata al primo site_profile, la seconda al secondo, etc. etc.
    Poi da controllare a manina...
    """
    sites_profile_PROBES = torch.all(
        (mask.int()[:, None, :] == site_profile_RESIDUES[None, :, :])
            .permute(1, 0, 2),
        dim=-1) # (#Profili x #Probes) #

    #print(sites_profile_PROBES.sum(dim=1))

    #print("sites_profile_PROBES ", sites_profile_PROBES.size())
    #print("site_profile_RESIDUES ", site_profile_RESIDUES.size())

    #distanza di grid_2d da tutti gli atomi della struttura

    # itero sui profili/siti identificati
    RES = {}
    for j in range(len(sites_profile_PROBES)):
        results = dict()

        # seleziono gli indici delle probes associate al profilo
        probe_idxs = sites_profile_PROBES[j].nonzero().view(-1)
        #print(probe_idxs)

        # seleziono le coordinate delle probes associate
        probe_xyz = grid_2d[probe_idxs]

        # prendo i residui associati al j-th profilo
        #profile_residues = site_profile_RESIDUES[j].nonzero().view(-1).tolist()

        residue_idxs = site_profile_RESIDUES[j].nonzero().view(-1).cpu()

        profile_residues_by_name = res_pos_chain[residue_idxs].tolist()
        #print(profile_residues_by_name, len(probe_idxs))

        n_res = len(profile_residues_by_name) #

        if all([x in predicted_res for x in profile_residues_by_name]):
            #print(profile_residues_by_name, len(probe_idxs), probe_xyz.mean(dim=0).cpu().numpy())
            if n_res not in RES.keys():
                RES[n_res] = []

            RES[n_res].append(
                #(profile_residues_by_name, len(probe_idxs), probe_xyz.mean(dim=0).cpu().numpy())
                {'residues':profile_residues_by_name, 'n_probes':len(probe_idxs), 'center':probe_xyz.mean(dim=0).cpu().numpy()}
            )


        else:
            print(profile_residues_by_name, len(probe_idxs), "NOP")

        #print("Probes CENTER", probe_xyz.mean(dim=0).cpu().numpy())
        #p = probe_xyz.mean(dim=0).view(1,-1)
        #distanza dall elemento più vicino ... ?
        #d = get_distances( coord_all, p).view(-1)
        #nz = (d<=1).float().nonzero().view(-1)
        #print("======>", nz)
        #print(d.size())

    del grid_2d, mask, site_profile_RESIDUES, sites_profile_PROBES, MASK_CA, MASK_CB, MASK_alpha, cos_alpha, prob_CA_distances, prob_CB_distances #


    def n_time_mean(xyz):
        avg = xyz.mean(dim=0)
        for _ in range(3):
            new_tensor = torch.cat((xyz, avg.unsqueeze(0)), dim=0)
            avg = new_tensor.mean(dim=0)
        return avg



    def last(_mask, _grid_2d):

        # todo - facciamo un clustering -> tanti cluster quanti siti

        #- prendo tutte le probes ok, non sono tante... sono quelle associate ai residui
        print(_mask.size())

        probes_ok_idxs = ((_mask > 0).int().sum(dim=1) > 2).int().nonzero().view(-1)
        coords_probe_ok = _grid_2d[probes_ok_idxs]
        #print(coords_probe_ok)
        print(coords_probe_ok.size())

        #- prendiamo il centro dell cluster ??
        #avg_coords = coords_probe_ok.mean(dim=0)
        avg_coords = n_time_mean(coords_probe_ok)
        print(avg_coords)

        # - prendiamo la probes reale più vicinaù
        for j in range(len(site_profile_RESIDUES)):
            probe_idxs = sites_profile_PROBES[j].nonzero().view(-1)
            probe_xyz = grid_2d[probe_idxs]

            residue_idxs = site_profile_RESIDUES[j].nonzero().view(-1).cpu()

            profile_residues_by_name = res_pos_chain[residue_idxs].tolist()

            dd = get_distances(avg_coords.unsqueeze(0), probe_xyz)
            print(profile_residues_by_name, dd.min().item())




        #mma = torch.any(mask > 0, dim=1)
        #print(mma.view(-1).size())
        #print(mma.int().nonzero().view(-1).size())

        #- prendiamo il sito predetto associato a quest ultima

    #last(mask, grid_2d)
    #sys.exit()

    return RES

#predict('12ca.pdb')
#predict('4O68.pdb')