import argparse
import uuid
import os.path as osp
import pickle
import torch
import torch.nn.functional as F
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikiCS
#from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from logger import Logger
from utils import reduce_train, augment_training_set, find_pseudolabels

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GAT(torch.nn.Module):
    def __init__(self, nfeats, hidden, nclasses, in_head, out_head,dropout=0.0):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_head = in_head
        self.out_head = out_head
        self.conv1 = GATConv(nfeats, hidden, heads=self.in_head)
        self.conv2 = GATConv(hidden*self.in_head, nclasses, concat=False,
                             heads=self.out_head)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x#F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, nfeats, hidden, nclasses,
                normalize=True, lin=True, dropout=0.0):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeats, hidden)
        self.conv2 = SAGEConv(hidden, nclasses)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x#F.log_softmax(x, dim=1)


def train(model, data, train_mask, optimizer, pseudo_ys):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    if pseudo_ys is not None:
        loss = F.cross_entropy(out[train_mask], pseudo_ys.squeeze()[train_mask])
    else:
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, train_mask, valid_mask, test_mask):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    #print('Test masl:  {}'.format(int(torch.count_nonzero(test_mask))))
    accs = []
    for mask in [train_mask, valid_mask, test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def train_loop(logger, split, device, data, train_mask, test_mask, val_mask, pseudo_ys, num_features, num_classes, args,  query, current_component):

    if args.method == 'GCN':
        model = GCN(num_features, args.hidden_channels, num_classes, args.dropout)
    elif args.method == 'SAGE':
        model = SAGE(num_features, args.hidden_channels, num_classes, args.dropout)
    elif args.method == 'GAT':
        model = GAT(num_features, args.hidden_channels, num_classes, args.in_head, args.out_head, args.dropout)

    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Only perform weight-decay on first convolution.


    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, train_mask, optimizer, pseudo_ys)
        result = test(model, data, train_mask, val_mask, test_mask)
        if current_component == 'AL':
            logger.add_al_result(split, query, result)
        elif current_component == 'ST':
            logger.add_st_result(split, query, result)
        else:
            logger.add_init_result(split, result)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {split + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--method', type=str, default='GCN')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--in_head', type=int, default=4)
    parser.add_argument('--out_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--splits', type=int, default=1)
    parser.add_argument('--use_AL', action='store_true')
    parser.add_argument('--AL_strategy', type=str, default='uncertainty')
    parser.add_argument('--use_ST', action='store_true')
    parser.add_argument('--num_pseudos', type=float, default=0.5)
    parser.add_argument('--stal_order', type=str, default='seq')
    parser.add_argument('--num_queries', type=int, default=2)
    args = parser.parse_args()
    print(args)

    uid = uuid.uuid4().hex

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid('./data/datasets/', args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    num_classes = dataset.num_classes
    print('The training set contains: {} classes.'.format(num_classes))
    num_features = data.num_features
    folder_path = "./data/splits/"+args.dataset+"/node/"

    #####
    num_labeled_instances_pretrain = 5*num_classes
    num_queries = 20*num_classes - num_labeled_instances_pretrain
    args.num_queries = num_queries
    #####
    logger = Logger(args.splits, args)

    data = data.to(device)

    for split_id in range(args.splits):

        train_mask = pickle.load(open(folder_path+args.dataset+"_train_mask_"+str(split_id), 'rb'))
        test_mask = pickle.load(open(folder_path+args.dataset+"_test_mask_"+str(split_id), 'rb'))
        val_mask = pickle.load(open(folder_path+args.dataset+"_val_mask_"+str(split_id), 'rb'))

        # max per class
        n_labeled = int(torch.count_nonzero(train_mask))
        max_perclass = int(n_labeled/num_classes)# 20 for citation networks

        pseudos_ys = None
        if args.use_AL:
            new_train_task = reduce_train(num_classes, n_labeled, train_mask)
            current_component = 'INIT'
            model = train_loop(logger, split_id, device, data, new_train_task, test_mask, val_mask, pseudos_ys, num_features, num_classes, args,  0, current_component)
            for query in range(num_queries):
                print('---------------Query: ', query)

                # get the query sample
                num_extra = 1
                new_train_mask, new_test_mask = augment_training_set(model, data, new_train_task, test_mask, num_extra, query, args)
                if args.stal_order == 'seq':
                    del model
                    # stal in two steps
                    # re-train with the augmented
                    current_component = 'AL'
                    #model = train_loop(device, run, split_idx, new_train_idx, data, dataset, logger, evaluator,  args, query, current_component)
                    print('AL learning with augmented {}'.format(int(torch.count_nonzero(new_train_mask))))
                    model = train_loop(logger, split_id, device, data, new_train_mask, new_test_mask, val_mask, pseudos_ys, num_features, num_classes, args,  query, current_component)
                    if args.use_ST:
                        # self-training with pseudo-edges
                        # first find best pseudo-edges
                        pseudo_train_mask, pseudos_ys = find_pseudolabels(model, data, new_train_mask, args)
                        del model
                        current_component = 'ST'
                        print('ST learning with augmented {}'.format(int(torch.count_nonzero(pseudo_train_mask))))
                        model = train_loop(logger, split_id, device, data, pseudo_train_mask, new_test_mask, val_mask, pseudos_ys, num_features, num_classes, args,  query, current_component)
                        pseudos_ys = None

        else:
            model = train_loop(logger, split_id, device, data, train_mask, test_mask, val_mask, pseudos_ys, num_features, num_classes, args, 0, 'bare')

        del model
        logger.add_run_result(split_id)
        #print('logger:', logger.results)


    logger.print_statistics()


if __name__ == "__main__":
    main()
