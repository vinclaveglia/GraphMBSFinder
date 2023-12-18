import os
from base import config
from base.PDB_MBS_preprocessing import get_dataset
from base.input import get_random_train_test_proteins
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys

# lista dei file embedding generati
embeddings = os.listdir(config.ProtT5_EMBEDDINGS_PATH)

# estrapolazione dei pdb code
proteins = [f"{x.split('.')[0]}.pdb" for x in embeddings]

# per avere la lista dei siti esistenti...
_, _, all_sites = get_random_train_test_proteins()

# generazione dei dataframe (con labels) associati ai pdb
print('get_dataset() ...')
DFs = get_dataset(proteins, all_sites, dataframe_only=True)



for _, df in DFs:
    df.reset_index(inplace=True)

layer = nn.Linear(in_features=1024, out_features=3)

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        H_SIZE = 100
        H_SIZE = 30
        self.fc1 = nn.Linear(in_features=1024, out_features=H_SIZE)
        self.fc2 = nn.Linear(in_features=H_SIZE, out_features=3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

layer = Classifier()

optimizer = optim.Adam(layer.parameters(), lr=0.001)

for epoch in range(100):
    tot_loss = 0.
    acc0_avg, acc1_avg, acc2_avg = [], [], []

    for h, k in DFs:
        optimizer.zero_grad()
        #print(h)
        pdb = h.split('.')[0]
        T5_path = config.ProtT5_EMBEDDINGS_PATH.joinpath(f"{pdb}.protT5.npy")
        X = torch.from_numpy(np.load(T5_path))
        X = X[:, :-1, :]

        labels = torch.tensor(k.target.to_numpy())

        #todo: gestire il fatto che potrebbero esserci 0 elementi di una certa classe ...

        C0_idxs = (labels == 0).nonzero().view(-1)
        C1_idxs = (labels == 1).nonzero().view(-1)
        C2_idxs = (labels == 2).nonzero().view(-1)

        if len(C0_idxs) == 0:
            print("[[no C0 idxs]]:", h)
            continue

        pred = layer(X).squeeze()

        idxs = (labels==2).nonzero()

        labels_oh = F.one_hot(labels)

        loss_C0 = ((pred[C0_idxs] - labels_oh[C0_idxs])** 2).mean()
        loss_C1 = ((pred[C1_idxs] - labels_oh[C1_idxs])** 2).mean()
        loss_C2 = ((pred[C2_idxs] - labels_oh[C2_idxs])** 2).mean()

        loss = loss_C0 + loss_C1 + loss_C2

        loss.backward()
        optimizer.step()

        tot_loss += loss.data

        acc0 = (pred[C0_idxs].argmax(dim=1) == 0).float().mean()
        acc1 = (pred[C1_idxs].argmax(dim=1) == 1).float().mean()
        acc2 = (pred[C2_idxs].argmax(dim=1) == 2).float().mean()
        #print(acc0, acc1, acc2)

        acc0_avg.append(acc0.item())
        acc1_avg.append(acc1.item())
        acc2_avg.append(acc2.item())

    acc0_avg = np.array(acc0_avg)
    acc1_avg = np.array(acc1_avg)
    acc2_avg = np.array(acc2_avg)

    print(acc0_avg.mean(), acc1_avg.mean(), acc2.mean())

    print(f"Epoch {epoch} => {tot_loss}")


#vedere se effettivamente il T5 aggiunge anche il token di fine stringa

#load the dataframe to extract the labels

#load the embedding

#create a simple linear model 1024 -> 50

#train the classifier

#1) rivedere il flusso dei dati per la versione [GCN + ProtT5] ... !!!

#2) prendere gli embedding di protT5 e la GCN, concatenarli e ri-classificare