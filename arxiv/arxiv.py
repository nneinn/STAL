####################################################################
#### code adapted from https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py

import argparse
import uuid
import torch
import pickle
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

from utils import augment_training_set, find_pseudolabels, reduce_train


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, nfeats, hidden, nclasses, in_head, out_head, dropout=0.0):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_head = in_head
        self.out_head = out_head
        self.bns = torch.nn.BatchNorm1d(hidden*self.in_head)
        self.conv1 = GATConv(nfeats, hidden, heads=self.in_head)
        self.conv2 = GATConv(hidden*self.in_head, nclasses, concat=False,
                             heads=self.out_head)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.bns.reset_parameters()

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, adj_t)
        x = self.bns(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer, pseudos_ys):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    if pseudos_ys is not None:
        loss = F.nll_loss(out, pseudos_ys.squeeze(1)[train_idx])
    else:
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train_loop(device, run, split_idx, train_idx, data, dataset, logger, evaluator, args, q, current_component, pseudos_ys):

    if args.method == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.method == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    elif args.method == 'GAT':
        model = GAT(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.in_head, args.out_head,
                    args.dropout).to(device)

    print(model)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer, pseudos_ys)
        result = test(model, data, split_idx, evaluator)
        if current_component == 'AL':
            logger.add_al_result(run, q, result)
        elif current_component == 'ST':
            logger.add_st_result(run, q, result)
        else:
            logger.add_init_result(run, result)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

    return model


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--method', type=str, default='GCN')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--in_head', type=int, default=4)
    parser.add_argument('--out_head', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use_AL', action='store_true')
    parser.add_argument('--AL_strategy', type=str, default='uncertainty')
    parser.add_argument('--num_queries', type=int, default=50)
    parser.add_argument('--pretrain_percentage', type=float, default=0.1)
    parser.add_argument('--AL_epochs', type=int, default=10)
    parser.add_argument('--use_ST', action='store_true')
    parser.add_argument('--num_pseudos', type=float, default=0.5)
    parser.add_argument('--stal_order', type=str, default='seq')
    args = parser.parse_args()
    print(args)

    uid = uuid.uuid4().hex

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    original_y = data.y.clone()

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    orig_test_idx = split_idx['test'].clone()#.to(device)

    print('classes:', dataset.num_classes)

    #####
    num_labeled_instances_pretrain = 5*dataset.num_classes
    num_queries = 20*dataset.num_classes - num_labeled_instances_pretrain
    args.num_queries = num_queries
    #####

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    n_labeled = int((train_idx.shape[0]))
    for run in range(args.runs):
        pseudos_ys = None
        split_idx['test'] = orig_test_idx
        if args.use_AL:
            current_component = 'INIT'

            # if we follow the 20C setup:#########
            new_train_idx, _ = reduce_train(data.x.shape[0], dataset.num_classes, n_labeled, train_idx)
            num_extra = 1
            #args.num_queries = num_queries
            ######################################

            model = train_loop(device, run, split_idx, new_train_idx, data, dataset, logger, evaluator,  args, 0, current_component, pseudos_ys)
            for query in range(args.num_queries):
                print('---------------Query: ', query)
                # get the augmented samples
                new_train_idx, new_test_idx = augment_training_set(model, data, split_idx['test'], new_train_idx, num_extra, query, args)

                if args.stal_order == 'seq':
                    del model
                    # stal in two steps
                    # re-train with the augmented
                    current_component = 'AL'
                    model = train_loop(device, run, split_idx, new_train_idx, data, dataset, logger, evaluator,  args, query, current_component, pseudos_ys)

                    if args.use_ST:
                        # self-training with pseudo-edges
                        # first find best pseudo-edges
                        pseudo_train_idx, pseudos_ys = find_pseudolabels(model, data, new_train_idx, args)
                        del model
                        current_component = 'ST'
                        model = train_loop(device, run, split_idx, pseudo_train_idx, data, dataset, logger, evaluator,  args, query, current_component, pseudos_ys)
                        pseudos_ys = None

        else:
            new_train_idx, _ = reduce_train_classes(data.x.shape[0], dataset.num_classes, n_labeled, train_idx)
            model = train_loop(device, run, split_idx, new_train_idx, data, dataset, logger, evaluator,  args, 0, 'bare', pseudos_ys)

        del model
        logger.add_run_result(run)

    logger.print_statistics()


if __name__ == "__main__":
    main()
