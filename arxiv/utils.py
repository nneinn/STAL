import torch
import numpy as np
import math
import networkx as nx
import torch.nn.functional as F

from numpy.random import choice
from torch_geometric.data import Data

from torch_geometric.utils import coalesce, dense_to_sparse, to_networkx
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

def reduce_train(xshape, num_classes, n_labeled, train_idx):
    train_mask = torch.zeros(xshape)
    train_mask[train_idx] = True
     # use 5 per class for pretrain
    num_labeled_instances_pretrain = 5*num_classes # fix this
    num_al_instances_per_query = 1 # for cora,citeseer, pubmed, we add only one new labeled training instance
    num_instances_for_al = n_labeled - num_labeled_instances_pretrain
    # a query for each instance of the rest of training
    num_queries = num_instances_for_al #int(n_labeled - num_labeled_instances_pretrain)

    idx = torch.where(train_mask == True)[0]
    num_to_change = num_instances_for_al
    idx_to_change = choice(idx, size=num_to_change, replace=False)
    train_mask[idx_to_change] = False
    train_idx = torch.where(train_mask==True)[0]

    return train_idx, num_queries

@torch.no_grad()
def augment_training_set(model, data, test_idx, train_idx, num_extra, query, args):
    # predict for all nodes # CAUTION costly!
    model.eval()
    out = model(data.x, data.adj_t)
    probs = F.log_softmax(out, dim=1)
    # get exp probs
    probs = torch.exp(probs)
    if args.AL_strategy == 'uncertainty':
        train_idx, test_idx = uncertainty_sample(probs, train_idx, test_idx, data, num_extra)
    elif args.AL_strategy == 'age':
        train_idx, test_idx = age_sample(probs, data, query, train_idx, test_idx, num_extra)
    elif args.AL_strategy == 'random':
        train_idx, test_idx = random_sample(probs, train_idx, test_idx, data, num_extra)
    else:
        assert('Wrong AL strategy!')

    return train_idx, test_idx

def append_new_label(train_mask, instance_to_add_idx):
    train_mask[instance_to_add_idx] = True
    return train_mask


def entropy_based(prob_dist, sorted=False):
    log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
    raw_entropy = 0 - torch.sum(log_probs, 1)
    raw_entropy = raw_entropy.unsqueeze(dim=1)

    normalized_entropy = raw_entropy / math.log2(prob_dist.numel())

    return normalized_entropy#.item()

def uncertainty_sample(probs, train_idx, test_idx, data, num_extra):
    train_mask = torch.zeros(data.x.shape[0])
    test_mask = torch.zeros(data.x.shape[0])
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    # get uncertainty scores
    marked = train_mask + test_mask
    uncertainty_scores = entropy_based(probs, sorted=False)

    unlabeled_indices = torch.where(marked==False)[0]
    unl_uncertain_scores = uncertainty_scores[unlabeled_indices]
    unl_most_uncertain = torch.topk(unl_uncertain_scores, num_extra, dim=0, largest=True)
    unl_most_uncertain_idx = unlabeled_indices[unl_most_uncertain[1]] # point to original indices
    # append most uncertain to training data
    train_mask = append_new_label(train_mask, unl_most_uncertain_idx)
    train_idx = torch.where(train_mask==True)[0]
    # remove new instance from test set
    test_mask[unl_most_uncertain_idx] = False
    test_idx = torch.where(test_mask==True)[0]
    return train_idx, test_idx

def random_sample(probs, train_idx, test_idx, data, num_extra):
    train_mask = torch.zeros(data.x.shape[0])
    test_mask = torch.zeros(data.x.shape[0])
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    # get the most uncertain unlabeled
    unlabeled_indices = torch.where(train_mask == False)[0]
    perm = torch.randperm(unlabeled_indices.size(0))
    idx = perm[:num_extra]
    samples = unlabeled_indices[idx]
    # append most uncertain to training data
    train_mask = append_new_label(train_mask, samples)
    # remove new instance from test set
    test_mask[samples] = False

    train_idx = torch.where(train_mask==True)[0]
    test_idx = torch.where(test_mask==True)[0]
    return train_idx, test_idx

@torch.no_grad()
def find_pseudolabels(model, data, train_idx, args):
    # predict for all nodes # CAUTION costly!
    model.eval()
    out = model(data.x, data.adj_t)
    # pick the top k nodes
    new_train_mask, new_labels = margin_based_pseudos(out, train_idx, data, args.num_pseudos)
    return new_train_mask, new_labels.unsqueeze(dim=1)


def margin_based_pseudos(embeddings, train_idx, data, per_pseudos):
    num_pseudos = int((data.x.shape[0]-train_idx.shape[0])*per_pseudos)
    pseudolabels = torch.argmax(embeddings, dim=1)
    pseudolabels_probs = torch.max(embeddings, dim=1)[0]
    train_mask = torch.zeros(data.x.shape[0])
    train_mask[train_idx] = True

    new_train_mask = train_mask.clone()
    unlabeled = torch.where(train_mask == False)[0]
    top2_values, top2_indices = torch.topk(embeddings, 2, dim=1, largest=True)
    top2_distance = (top2_values[:,0].unsqueeze(dim=1) - top2_values[:,1].unsqueeze(dim=1)).pow(2).sqrt()
    top2_dist_unlabeld = top2_distance[unlabeled]
    best_pseudos_val, best_pseudos_ind = torch.topk(top2_dist_unlabeld, num_pseudos, dim=0, largest=True)
    best_unlabeled = unlabeled[best_pseudos_ind.squeeze()]
    new_train_mask[best_unlabeled] = True
    ll2 = pseudolabels[best_unlabeled]
    new_labels = data.y.clone()
    if len(new_labels.shape)>1: # check for proper dims in case of ogbn data
        new_labels = new_labels.squeeze()
    #print('new_labels: ', new_labels.shape)
    new_labels[best_unlabeled] = ll2
    # use idxs instead
    new_train_idx = torch.where(new_train_mask==True)[0]
    return new_train_idx, new_labels


def age_sample(probs, data, q, train_idx, test_idx, num_points):
    train_mask = torch.zeros(data.x.shape[0])
    train_mask[train_idx] = True

    test_mask = torch.zeros(data.x.shape[0])
    test_mask[test_idx] = True

    row,col,_ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)
    data_pyg = Data(edge_index=edge_index, num_nodes=data.x.shape[0])
    nxG = to_networkx(data_pyg)
    del data_pyg
    #print('nx')
    normcen = centralissimo(nxG).flatten()
    #print('normcen', len(normcen))
    cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])
    #print('cenperc')
    n_ks = len(np.unique(data.y.cpu().numpy()))
    basef = 0.995
    gamma = np.random.beta(1, 1.005-basef**q)
    alpha = beta = (1-gamma)/2
    unlabeled_indices = torch.where(train_mask == False)[0]
    # get uncertainty scores
    scores = torch.sum(-F.softmax(probs, dim=1) * F.log_softmax(probs, dim=1), dim=1)
    softmax_out = F.softmax(probs, dim=1).cpu().detach().numpy()

    #print('entrperc')
    entrperc = perc_full_np(scores.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=n_ks, random_state=0).fit(softmax_out)
    ed=euclidean_distances(softmax_out,kmeans.cluster_centers_)
    ed_score = np.min(ed,axis=1)
    edprec = 1. - perc_full_np(ed_score)
    finalweight = alpha*entrperc + beta*edprec + gamma*cenperc

    unl_uncertain_scores = finalweight[unlabeled_indices]
    unl_most_uncertain = np.argsort(unl_uncertain_scores)[::-1][:num_points]
    unl_most_uncertain_idx = unlabeled_indices[unl_most_uncertain[0]]
    # append most uncertain to training data
    train_mask = append_new_label(train_mask, unl_most_uncertain_idx)
    # remove new instance from test set
    test_mask[unl_most_uncertain_idx] = False

    new_train_idx = torch.where(train_mask==True)[0]
    new_test_idx = torch.where(test_mask==True)[0]

    return new_train_idx, new_test_idx

#################################
## AGE functions
#############################
def centralissimo(G):
    ## the code is adapted from http://github.com/xxxx
    centralities = []
    centralities.append(nx.pagerank(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
    	cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen

#calculate the percentage of elements smaller than the k-th element
def perc(input,k):
    ## the code is adapted from https://github.com/vwz/AGE
    sm = input[input<input[k]]
    per = sm.shape[0]/input.shape[0]
    return per

#calculate the percentage of elements larger than the k-th element
def percd(input,k):
    ## the code below is adapted from https://github.com/vwz/AGE
    sm = input[input>input[k]]
    per = sm.shape[0]/input.shape[0]
    return per

def perc_full_np(input):
    ## the code below is adapted from https://github.com/vwz/AGE
    l = len(input)
    indices = np.argsort(input)
    loc = np.zeros(l, dtype=np.float)
    for i in range(l):
        loc[indices[i]] = i
    return loc / l
