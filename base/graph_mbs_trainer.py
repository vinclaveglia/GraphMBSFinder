import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch_geometric
from base.PDB_MBS_preprocessing import get_dataset
from base.model import ProteinSegmenter2
from base.input import get_random_train_test_proteins
from base.evaluation import evaluate
from base.monitor import Monitor
import warnings

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_pdb, test_pdb, db_sites = get_random_train_test_proteins()
train_pdb = train_pdb[:100]
test_pdb = test_pdb[:30]

train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

GRAPH_DATASET = get_dataset(train_pdb+test_pdb, db_sites) # prende in input la site list

TRAIN_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in train_pdb if pdb in list(GRAPH_DATASET.keys())]
TEST_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in test_pdb if pdb in list(GRAPH_DATASET.keys())]

for x in TEST_GRAPH_DATASET:
    x.to(device)
for x in TRAIN_GRAPH_DATASET:
    x.to(device)

my_loader_train = DataLoader(TRAIN_GRAPH_DATASET, batch_size=16, shuffle=True)
my_loader_test = DataLoader(TEST_GRAPH_DATASET, batch_size=16, shuffle=True)

######################
# QUI SI PUO INIZIARE IL GRID-RANDOM SEARCH

'''
Cosa facciamo variare ???
Dobbiamo stampare e memorizzare un bel po di robe
 
'''

net2 = ProteinSegmenter2(num_node_features=22, num_classes=3)
net2.to(device)

optim2 = torch.optim.Adam(net2.parameters(), lr=0.01)

max_epoch = 200

train_monitor, test_monitor = Monitor(), Monitor()

# Training
for epoch in range(max_epoch):
    # TRAIN
    for jj, batch in enumerate(my_loader_train):
        batch.to(device)
        #print(jj, len(my_loader_train))
        aaa = torch_geometric.utils.to_dense_adj(batch.edge_index)[0]
        net2.zero_grad()
        pred = net2(batch)
        avg_loss, performance = evaluate(pred, batch.y.argmax(dim=1), aaa, epoch)
        avg_loss.backward()
        optim2.step()
        with torch.no_grad():
            train_monitor.store(performance)

    # TEST
    if epoch % 5 == 0:
        for batch_tst in my_loader_test:
            batch_tst.to(device)
            aaa = torch_geometric.utils.to_dense_adj(batch_tst.edge_index)[0]
            with torch.no_grad():
                pred_tst = net2(batch_tst)
                avg_loss_test, performance_test = evaluate(pred_tst, batch_tst.y.argmax(dim=1), aaa, epoch)
                test_monitor.store(performance_test)

        print(f"Epoch {epoch} train loss: ",
              train_monitor.get_epoch_values('tot_loss', epoch),
              test_monitor.get_epoch_values('tot_loss', epoch),
              train_monitor.get_epoch_values('FP', epoch),
              test_monitor.get_epoch_values('FP', epoch),
              #train_monitor.get_epoch_values('acc2', epoch),
              test_monitor.get_epoch_values('acc2', epoch))


train_monitor.finalize()
test_monitor.finalize()

import matplotlib.pyplot as plt
plt.plot(train_monitor.get_values('tot_loss'), label="train loss")
plt.plot(test_monitor.get_values('tot_loss'), label="test loss")
plt.legend()
plt.show()

plt.plot(train_monitor.get_values('loss_group'), label="group train")
plt.plot(test_monitor.get_values('loss_group'), label="group test")
plt.legend()
plt.show()


for test_graph in TRAIN_GRAPH_DATASET[:20]:

    print(test_graph)

    pred = net2(test_graph)

    residues = test_graph['residues']

    CA_br2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)
    CA_br2_idxs_T = (test_graph.y.argmax(dim=1) == 2).nonzero().view(-1)

    CA_br1_idxs = (pred.argmax(dim=1) == 1).nonzero().view(-1)
    CA_br1_idxs_T = (test_graph.y.argmax(dim=1) == 1).nonzero().view(-1)

    print('BR2',[residues[i] for i in CA_br2_idxs])
    print('BR2',[residues[i] for i in CA_br2_idxs_T])
    print("----------------------------------")
    #print('BR1',[residues[i] for i in CA_br1_idxs])
    #print('BR1',[residues[i] for i in CA_br1_idxs_T])

print("%%%"*30)
for test_graph in TEST_GRAPH_DATASET:

    fprl, fprl2 = [], []



    pred = net2(test_graph)

    residues = test_graph['residues']

    CA_br2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)
    CA_br2_idxs_T = (test_graph.y.argmax(dim=1) == 2).nonzero().view(-1)

    CA_br1_idxs = (pred.argmax(dim=1) == 1).nonzero().view(-1)
    CA_br1_idxs_T = (test_graph.y.argmax(dim=1) == 1).nonzero().view(-1)

    NON_2_idxs = (test_graph.y.argmax(dim=1) != 2).nonzero().view(-1)
    FP_idxs = (pred[NON_2_idxs].argmax(dim=1) == 2).nonzero().view(-1)
    FPR = np.round(len(FP_idxs)*100/len(CA_br2_idxs_T), 2)
    fprl.append(FPR)

    tt = torch.softmax(pred, dim=1).cpu()
    #print(tt[CA_br2_idxs])
    pl = tt[:, 2]

    PRED = [residues[i] for i in CA_br2_idxs]
    TARGET = [residues[i] for i in CA_br2_idxs_T]

    print(test_graph, f'[[ FPR {int(FPR)}%]]')
    print('BR2',PRED)
    print('BR2',TARGET)
    #print([np.round(pl[i].data.item(), 4) for i in CA_br2_idxs])
    print("----------------------------------")
    #print('BR1',[residues[i] for i in CA_br1_idxs])
    #print('BR1',[residues[i] for i in CA_br1_idxs_T])

print(np.mean(fprl))