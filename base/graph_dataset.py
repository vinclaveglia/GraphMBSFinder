from torch_geometric.data import Dataset
import os.path as osp
import os
from base.input import get_graph

"""
def create_graph_object(features, adj_matrix, node_target=None, df= None, name = "prova"):

    edge_index = adj_matrix.nonzero().t().contiguous().long()

    x = features

    if node_target != None:
        data = Data(x=x, edge_index=edge_index, y=node_target)
    else:
        data = Data(x=x, edge_index=edge_index)

    data["name"] = name
    data['residues'] = df['res_pos_chain'].tolist()

    return data

def get_graph(prot):

    features = np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.pdb.feat'))
    features = torch.from_numpy(features)

    distances = np.loadtxt(os.path.join(config.PROTS_PATH, f'{prot}.pdb.dist'))
    distances = torch.from_numpy(distances)

    df = pd.read_csv(os.path.join(config.PROTS_PATH, f'{prot}.pdb.labels'))

    target = torch.tensor(df['target'])
    target_oh = F.one_hot(target, num_classes=3)

    graph = create_graph_object(features, adj_matrix=distances, node_target=target_oh, name=prot, df=df)

    return graph
"""

class ProtGraphDataset(Dataset):

    #RAW_DIR # dati grezzi
    #PROCESSED_DIR # dati processati

    """
    - TRANSFORM: dynamically transforms the data object before accessing (so it is best used for data augmentation)
    
    - PRE_TRANSFORM: function applies the transformation before saving the data objects to disk (so it is best used 
    for heavy precomputation which needs to be only done once).
    
    - PRE_FILTER: can manually filter out data objects before saving. Use cases may involve the restriction 
    of data objects being of a specific class. 
    """

    def __init__(self, root, prot_names,
                 pdb_dir = 'proteins', sites_dir = 'sites',
                 transform=None, pre_transform=None, pre_filter=None):

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.proteins_path = osp.join(root, pdb_dir)
        self.sites_path = osp.join(root, sites_dir)

        self.prot_names = []
        print("creating ...")
        for p in prot_names:
            if os.path.exists(osp.join(self.proteins_path, f'{p}.pdb.dist')):
                self.prot_names.append(p)
        #self.site_names = site_names
        print("prot dataset created")

    """@property
    def processed_file_names(self):
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
        return [x for x in os.listdir(self.processed_dir)] """


    """ def process(self):
        # Processes raw data and saves it into the processed_dir.
        [x for x in os.listdir(self.r)]
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1 """


    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        data = get_graph(self.prot_names[idx])
        return data

    def len(self):
        return len(self.prot_names)

#prots = ['4ahb', '1xrf', '5xkq', '2jvy', '2bco']

#prot_dataset = ProtGraphDataset('../data', prot_names=prots)
#print(prot_dataset[0])
#print(prot_dataset[1])


