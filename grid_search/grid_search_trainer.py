import sys, os
root_path = os.path.join(
    os.path.dirname(__file__), os.pardir)
sys.path.insert(0, root_path)
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch_geometric


from base.PDB_MBS_preprocessing import get_dataset
from base.wp_model import ProteinSegmenter2
from base.input import get_random_train_test_proteins
from base.evaluation import evaluate
from base.monitor import Monitor
from base.config import Atom
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('__file__:    ', __file__)
import pathlib
print(pathlib.Path(__file__).parent)
print(os.path.dirname(__file__))

train_pdb, test_pdb, db_sites = get_random_train_test_proteins()

#train_pdb = train_pdb[:3]
#test_pdb = test_pdb[:3]
#test_pdb = []

train_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in train_pdb]
test_sites = [x for x in db_sites if f"{x.split('_')[0]}.pdb" in test_pdb]

# --- rilanciare sta roba ma con i CB --- #
print("Data loading...")
GRAPH_DATASET = get_dataset(train_pdb+test_pdb, Atom.CB, db_sites) # prende in input la site list
#GRAPH_DATASET = get_dataset(test_pdb, Atom.CB, db_sites) # prende in input la site list
print("fine get dataset")

TRAIN_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in train_pdb if pdb in list(GRAPH_DATASET.keys())]
TEST_GRAPH_DATASET = [GRAPH_DATASET[pdb] for pdb in test_pdb if pdb in list(GRAPH_DATASET.keys())]

for x in TEST_GRAPH_DATASET:
    x.to(device)
for x in TRAIN_GRAPH_DATASET:
    x.to(device)


my_loader_train = DataLoader(TRAIN_GRAPH_DATASET, batch_size=32, shuffle=True)
my_loader_test = DataLoader(TEST_GRAPH_DATASET, batch_size=32, shuffle=True)

######################################################################################

max_epoch = 101

config_runs = pd.DataFrame()

n_init = 3
layer_size = [50, 80, 100]
lambda_C0 = [1, 2, 5, 7]
lambda_group = [0, 1, 3, 5]
learning_rates = [0.01, 0.001, 0.1]
ITER = 0

lambda_C0 = [2]
lambda_group = [5]
learning_rates = [0.01]
layer_size = [50]

#- al di la dello storage dell andamento del training
#- ci vuole un diario che raccolga i risultati di tutti i vari
#tentativi di training con le diverse configurazioni, che è quello che conta di più

training_diary = Monitor()

for ls in layer_size:

    for lc0 in lambda_C0:

        for lgroup in lambda_group:

            for lr in learning_rates:

                ITER += 1
                for init in range(n_init):
                    print('CONFIG: Iter',ITER, 'init', init, 'lambda C0', lc0, 'layer_size', ls, 'lambda_groud', lgroup, 'lr', lr)
                    net2 = ProteinSegmenter2(num_node_features=22, num_classes=3, layer_size = ls)
                    net2.to(device)
                    optim2 = torch.optim.Adam(net2.parameters(), lr=lr)

                    train_monitor, test_monitor = Monitor(), Monitor()

                    best_epoch = 0
                    best_loss = float('inf')
                    best_performance = {}

                    # Training
                    for epoch in range(max_epoch):
                        # TRAIN
                        for jj, batch in enumerate(my_loader_train):
                            batch.to(device)
                            #print(jj, len(my_loader_train))
                            aaa = torch_geometric.utils.to_dense_adj(batch.edge_index)[0]
                            net2.zero_grad()
                            pred = net2(batch)
                            avg_loss, performance = evaluate(pred, batch.y.argmax(dim=1), aaa, epoch, lambda_C0=lc0, lambda_group=lgroup)
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
                                    avg_loss_test, performance_test = evaluate(pred_tst, batch_tst.y.argmax(dim=1), aaa, epoch, lambda_C0=lc0, lambda_group=lgroup)
                                    test_monitor.store(performance_test)
                                    #print(avg_loss_test)
                                    if avg_loss_test < best_loss:
                                        torch.save(net2.state_dict(), f'model_{lr}_{ls}_{lc0}_{lgroup}_{init}.CB25ott.pth')
                                        best_epoch = epoch
                                        best_loss = avg_loss_test
                                        best_performance = performance_test

                        #... ficcarci il weight delle connessioni ...

                    train_monitor.finalize()
                    test_monitor.finalize()

                    print(best_performance)

                    if len(best_performance) > 0:
                        to_store = {
                            'run_id':f'R_{ITER}',
                            **best_performance,
                            'lambda_C0':lc0,
                            'layer_size':ls,
                            'lambda_group':lgroup,
                            'lr':lr
                        }
                        training_diary.store(to_store)
                    else:
                        print("- Non numerical values -")

            #training_diary.save("../../diary_cp.csv")
            training_diary.save("diary_tmp_25ott.csv")

            # todo: mettere sta roba in un notebook please