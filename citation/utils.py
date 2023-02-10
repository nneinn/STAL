import torch
import numpy as np
import math
import networkx as nx
import torch.nn.functional as F

from numpy.random import choice
from torch_geometric.utils import coalesce, dense_to_sparse, to_networkx
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

def reduce_train(num_classes, n_labeled, train_mask):
     # use 5 per class for pretrain
    np.random.seed(123456)
    num_labeled_instances_pretrain = 5*num_classes
    num_al_instances_per_query = 1 # for cora,citeseer, pubmed, we add only one new labeled training instance
    num_instances_for_al = n_labeled - num_labeled_instances_pretrain
    # a query for each instance of the rest of training
    num_queries = num_instances_for_al #int(n_labeled - num_labeled_instances_pretrain)

    idx = torch.where(train_mask == True)[0]
    num_to_change = num_instances_for_al
    idx_to_change = choice(idx, size=num_to_change, replace=False)
    #print(idx_to_change)
    train_mask[idx_to_change] = False

    return train_mask


@torch.no_grad()
def find_pseudolabels(model, data, train_mask, args):
    # predict for all nodes # CAUTION costly!
    model.eval()
    out = model(data.x, data.edge_index)
    # pick the top k nodes
    new_train_mask, new_labels = margin_based_pseudos(out, train_mask, data.y, args.num_pseudos)
    return new_train_mask, new_labels.unsqueeze(dim=1)


def margin_based_pseudos(embeddings, train_mask, ys, per_pseudos):
    pseudolabels = torch.argmax(embeddings, dim=1)
    pseudolabels_probs = torch.max(embeddings, dim=1)[0]
    new_train_mask = train_mask.clone()
    unlabeled = torch.where(train_mask == False)[0]
    num_pseudos = int((unlabeled.shape[0]*per_pseudos))
    top2_values, top2_indices = torch.topk(embeddings, 2, dim=1, largest=True)
    top2_distance = (top2_values[:,0].unsqueeze(dim=1) - top2_values[:,1].unsqueeze(dim=1)).pow(2).sqrt()
    top2_dist_unlabeld = top2_distance[unlabeled]
    best_pseudos_val, best_pseudos_ind = torch.topk(top2_dist_unlabeld, num_pseudos, dim=0, largest=True)
    best_unlabeled = unlabeled[best_pseudos_ind.squeeze()]
    new_train_mask[best_unlabeled] = True
    ll2 = pseudolabels[best_unlabeled]
    new_labels = ys.clone()
    if len(new_labels.shape)>1: # check for proper dims in case of ogbn data
        new_labels = new_labels.squeeze()
    new_labels[best_unlabeled] = ll2
    return new_train_mask, new_labels

@torch.no_grad()
def augment_training_set(model, data, train_mask, test_mask, num_extra, query, args):
    # we need to get all unlabeled nodes
    # predict for all nodes # CAUTION costly!
    model.eval()
    out = model(data.x, data.edge_index)
    probs = F.log_softmax(out, dim=1)
    # get exp probs
    probs = torch.exp(probs)
    if args.AL_strategy == 'uncertainty':
        train_mask, test_mask = uncertainty_sample(probs, train_mask, test_mask, num_extra)
    elif args.AL_strategy == 'age':
        train_mask, test_mask = age_sample(probs, data, query, train_mask, test_mask, num_extra)
    else:
        assert('Wrong AL strategy!')

    return train_mask, test_mask


def entropy_based(prob_dist, sorted=False):
    log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
    raw_entropy = 0 - torch.sum(log_probs, 1)
    raw_entropy = raw_entropy.unsqueeze(dim=1)
    normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
    return normalized_entropy#.item()

def append_new_label(train_mask, instance_to_add_idx):
    train_mask[instance_to_add_idx] = True
    return train_mask


def uncertainty_sample(probs, train_mask, test_mask, num_extra):
    # get uncertainty scores
    uncertainty_scores = entropy_based(probs, sorted=False)
    # get the most uncertain unlabeled
    unlabeled_indices = torch.where(train_mask == False)[0]
    unl_uncertain_scores = uncertainty_scores[unlabeled_indices]
    unl_most_uncertain = torch.topk(unl_uncertain_scores, num_extra, dim=0, largest=True)

    unl_most_uncertain_idx = unlabeled_indices[unl_most_uncertain[1]] # point to original indices
    # append most uncertain to training data
    train_mask = append_new_label(train_mask, unl_most_uncertain_idx)
    # remove new instance from test set
    test_mask[unl_most_uncertain_idx] = False
    return train_mask, test_mask

def age_sample(probs, data, q, train_mask, test_mask, num_points):
    nxG = to_networkx(data)
    normcen = centralissimo(nxG).flatten()

    cenperc = np.asarray([perc(normcen,i) for i in range(len(normcen))])
    n_ks = len(np.unique(data.y.cpu().numpy()))
    basef = 0.995
    gamma = np.random.beta(1, 1.005-basef**q)
    alpha = beta = (1-gamma)/2
    unlabeled_indices = torch.where(train_mask == False)[0]
    # get uncertainty scores
    scores = torch.sum(-F.softmax(probs, dim=1) * F.log_softmax(probs, dim=1), dim=1)
    softmax_out = F.softmax(probs, dim=1).cpu().detach().numpy()

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
    return train_mask, test_mask

#################################
## AGE functions
#############################
def centralissimo(G):
    ## the code is adapted from https://github.com/vwz/AGE
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
    return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

#calculate the percentage of elements larger than the k-th element
def percd(input,k):
    ## the code below is adapted from https://github.com/vwz/AGE
    return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

def perc_full_np(input):
    ## the code below is adapted from https://github.com/vwz/AGE
    l = len(input)
    indices = np.argsort(input)
    loc = np.zeros(l, dtype=np.float)
    for i in range(l):
        loc[indices[i]] = i
    return loc / l
