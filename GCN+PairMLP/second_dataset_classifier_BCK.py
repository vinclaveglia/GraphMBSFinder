import pandas as pd
import numpy as np
import sys
import random
import pandas as pd
from ast import literal_eval

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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



def get_data(data_file):

    data = pd.read_csv(data_file, index_col=None)

    Y = torch.from_numpy(data['target'].astype(float).to_numpy())

    _C0_idxs = (Y == 0).nonzero().view(-1)
    _C1_idxs = (Y == 1).nonzero().view(-1)

    Y_oh = torch.zeros(len(Y), 2)
    Y_oh[_C0_idxs,0] = 1
    Y_oh[_C1_idxs,1] = 1
    Y_oh = Y_oh.to(device)

    data['x1'] = data['x1'].apply(literal_eval)
    X1 = np.array(data['x1'].tolist())

    data['x2'] = data['x2'].apply(literal_eval)
    X2 = np.array(data['x2'].tolist())

    data['rel'] = data['rel'].apply(literal_eval)
    Rel = torch.from_numpy(np.array(data['rel'].tolist()))
    Rel.to(device)

    XX = torch.from_numpy(np.concatenate((X1, X2),axis=1))
    XX.to(device)

    Rel = Rel.float()
    Rel = Rel.to(device)

    XX = XX.float()
    XX = XX.to(device)

    return XX, Rel, Y_oh


def evaluate(pred, target, net):
    c0_idxs = (target.argmax(dim=1) == 0).nonzero().view(-1)
    c1_idxs = (target.argmax(dim=1) == 1).nonzero().view(-1)
    acc0 = (pred[c0_idxs].argmax(dim=1) == target[c0_idxs].argmax(dim=1)).float().mean()
    acc1 = (pred[c1_idxs].argmax(dim=1) == target[c1_idxs].argmax(dim=1)).float().mean()
    loss = ((pred - target)**2).mean()
    return loss, acc0, acc1




XX_train, Rel_train, Y_oh_train = get_data('seconda_dataset_ALL.csv')
XX_test, Rel_test, Y_oh_test = get_data('seconda_dataset_TEST.csv')
C0_idx_test = (Y_oh_test.argmax(dim=1) == 0).nonzero().view(-1)
C1_idx_test = (Y_oh_test.argmax(dim=1) == 1).nonzero().view(-1)


pair_net = Pair_mlp()
pair_net.to(device)
optim = torch.optim.Adam(params=pair_net.parameters(), lr = 0.0001)


for epoch in range(100000):
    optim.zero_grad()
    rnd_idxs = random.sample(range(0, len(data)), 256)

    xx, rel, yy = XX_train[rnd_idxs], Rel_train[rnd_idxs], Y_oh_train[rnd_idxs]

    _c0_idxs = (yy.argmax(dim=1) == 0).nonzero().view(-1)
    _c1_idxs = (yy.argmax(dim=1) == 1).nonzero().view(-1)

    pred = pair_net(xx, rel)
    loss = ((pred - yy)**2).mean()
    loss.backward()
    optim.step()

    acc0 = (pred[_c0_idxs].argmax(dim=1) == yy[_c0_idxs].argmax(dim=1)).float().mean()
    acc1 = (pred[_c1_idxs].argmax(dim=1) == yy[_c1_idxs].argmax(dim=1)).float().mean()

    if epoch % 50 == 0:
        with torch.no_grad():
            pred_test = pair_net(XX_test, Rel_test)
            acc0_test = (pred_test[C0_idx_test].argmax(dim=1) == Y_oh_test[C0_idx_test].argmax(dim=1)).float().mean()
            acc1 = (pred[_c1_idxs].argmax(dim=1) == yy[_c1_idxs].argmax(dim=1)).float().mean()

            print(f"Epoch {epoch}",round(loss.data.item(),8), round(acc0.item(), 4), round(acc1.item(), 4))
