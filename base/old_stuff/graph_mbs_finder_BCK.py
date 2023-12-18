import torch
from torch_geometric.loader import DataLoader
import torch_geometric
from base.PDB_MBS_preprocessing import get_dataset
from base.wp_model import ProteinSegmenter2
from base.input import get_train_test

# Input processing

#_sites = ['4ahb_2', '1xrf_1', '5xkq_2', '2jvy_1', '2bco_1',
#         '3oru_1', '4ywy_2', '5jwl_1', '4hcc_2', '3rpd_1',
#         '2hf8_5', '5chj_6', '5vwm_1']

#_sites = get_test_sites()
train, test = get_train_test()

TRAIN_GRAPH_DATASET = get_dataset(train[:50])
TEST_GRAPH_DATASET = get_dataset(train[:10])

my_loader_train = DataLoader(TRAIN_GRAPH_DATASET, batch_size=10, shuffle=True)
my_loader_test = DataLoader(TEST_GRAPH_DATASET, batch_size=1, shuffle=True)

#net = ProteinSegmenter(nfeat=5, n_h=30, n_out=3, dropout=0.)
#net2 = ProteinSegmenter2(num_node_features=5, num_classes=3)
net2 = ProteinSegmenter2(num_node_features=22, num_classes=3)

#optim = torch.optim.Adam(net.parameters(), lr=0.01)
optim2 = torch.optim.Adam(net2.parameters(), lr=0.01)

max_epoch = 50 #300

#loss_fn = torch.nn.CrossEntropyLoss()

#plottare un po di cazzo di grafici di training...

loss_time = []
loss_group_time = []

# Training
for epoch in range(max_epoch):

    print("%%"*30, epoch)

    epoch_loss = 0.
    epoch_loss_group = 0.

    for batch in my_loader_train:

        aaa = torch_geometric.utils.to_dense_adj(batch.edge_index)[0]

        net2.zero_grad()

        pred = net2(batch)

        pred_br2_idxs = (pred.argmax(dim=1)==2).nonzero().view(-1)

        #controllare nella matrice di adiacenza che i nodi predetti come br2 abbiano almeno 2 vicini predetti come br2

        aaabr2 = aaa[pred_br2_idxs, :][:, pred_br2_idxs]
        n_vicini_tipo_br2 = aaabr2.sum(dim=1)

        #loss_group = torch.abs(n_vicini_tipo_br2 - 2)
        loss_group = (n_vicini_tipo_br2 < 2).float().mean()

        target_onehot = batch.y
        target = target_onehot.argmax(dim=1)

        CA_br2_idxs = (target == 2).nonzero().view(-1)
        CA_br1_idxs = (target == 1).nonzero().view(-1)
        CA_br0_idxs = (target == 0).nonzero().view(-1)

        avg_loss0 = loss_fn(pred[CA_br0_idxs], target[CA_br0_idxs])
        avg_loss1 = loss_fn(pred[CA_br1_idxs], target[CA_br1_idxs])
        avg_loss2 = loss_fn(pred[CA_br2_idxs], target[CA_br2_idxs])

        avg_loss = 2*avg_loss0 + avg_loss1 + avg_loss2
        if epoch > 100:
            avg_loss += 5*loss_group


        avg_loss.backward()
        epoch_loss += avg_loss.data.item()
        epoch_loss_group += loss_group.data.item()

        optim2.step()

        #print(avg_loss0.item(), avg_loss1.item(), avg_loss2.item())

        acc0 = (pred[CA_br0_idxs].argmax(dim=1) == target[CA_br0_idxs]).float().mean().item()
        acc1 = (pred[CA_br1_idxs].argmax(dim=1) == target[CA_br1_idxs]).float().mean().item()
        acc2 = (pred[CA_br2_idxs].argmax(dim=1) == target[CA_br2_idxs]).float().mean().item()

        proteins = batch['name']
        res = {'proteins': proteins, 'epoch': epoch, 'acc0': acc0, 'acc1': acc1, 'acc2': acc2}
        #print(res)
    print("Epoc loss: ", epoch_loss)
    loss_time.append(epoch_loss)
    loss_group_time.append(epoch_loss_group)

import matplotlib.pyplot as plt
plt.plot(loss_time, label="avg_loss")
plt.legend()
plt.show()

plt.plot(loss_group_time, label="group")
plt.legend()
plt.show()

for batch in my_loader_test:
    print("/////////////////////////////////////////////////////////////", batch['name'])
    residues = batch['residues'][0]
    pred = net2(batch)
    CA_br2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)
    CA_br2_idxs_T = (batch.y.argmax(dim=1) == 2).nonzero().view(-1)

    CA_br1_idxs = (pred.argmax(dim=1) == 1).nonzero().view(-1)
    CA_br1_idxs_T = (batch.y.argmax(dim=1) == 1).nonzero().view(-1)

    print('BR2',[residues[i] for i in CA_br2_idxs])
    print('BR2',[residues[i] for i in CA_br2_idxs_T])
    print("----------------------------------")
    #print('BR1',[residues[i] for i in CA_br1_idxs])
    #print('BR1',[residues[i] for i in CA_br1_idxs_T])