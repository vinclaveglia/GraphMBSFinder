import numpy as np
import torch
#prova

def evaluate(pred, target, adj, epoch, lambda_C0, lambda_group):

    loss_fn = torch.nn.CrossEntropyLoss()

    pred_br2_idxs = (pred.argmax(dim=1) == 2).nonzero().view(-1)

    # controllare nella matrice di adiacenza che i nodi predetti come br2 abbiano almeno 2 vicini predetti come br2
    aaabr2 = adj[pred_br2_idxs, :][:, pred_br2_idxs]
    n_vicini_tipo_br2 = aaabr2.sum(dim=1)

    CA_br2_idxs = (target == 2).nonzero().view(-1)
    CA_br1_idxs = (target == 1).nonzero().view(-1)
    CA_br0_idxs = (target == 0).nonzero().view(-1)
    target_NON_2_idxs = torch.cat((CA_br0_idxs, CA_br1_idxs))

    avg_loss0 = loss_fn(pred[CA_br0_idxs], target[CA_br0_idxs])
    avg_loss1 = loss_fn(pred[CA_br1_idxs], target[CA_br1_idxs])
    avg_loss2 = loss_fn(pred[CA_br2_idxs], target[CA_br2_idxs])

    # la loss su C0 va tenuta a bada
    CE_loss = lambda_C0 * avg_loss0 + avg_loss1 + avg_loss2

    loss_group = lambda_group * (n_vicini_tipo_br2 < 2).float().mean()

    tot_loss = CE_loss + loss_group


    acc0 = (pred[CA_br0_idxs].argmax(dim=1) == target[CA_br0_idxs]).float().mean().item()
    acc1 = (pred[CA_br1_idxs].argmax(dim=1) == target[CA_br1_idxs]).float().mean().item()
    acc2 = (pred[CA_br2_idxs].argmax(dim=1) == target[CA_br2_idxs]).float().mean().item()


    n_C2 = len(CA_br2_idxs)

    false_positive = (pred[target_NON_2_idxs].argmax(dim=1) == 2).float().sum().item() / n_C2

    #return tot_loss, CE_loss, loss_group
    return tot_loss, {'tot_loss': tot_loss.data.item(),
                      'CE_loss':CE_loss.data.item(),
                      'loss_group': loss_group.data.item(),
                      'epoch':epoch,
                      'acc0': np.round(acc0, 3),
                      'acc1': np.round(acc1, 3),
                      'acc2': np.round(acc2, 3),
                      'FP':np.round(false_positive,2)*100
                      }

###