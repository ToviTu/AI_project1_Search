from graphLib import Graph
from scipy.sparse import linalg
import scipy as sp
import scipy.sparse as ssp
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv

# =================== Task 3: link prediction ===================


# generate an observed network
def gen_net_obs(graph, test_edges, test_non_edges):
    adj_matrix = graph.get_adj_matrix()
    # adjacency matrix of the observed graph, which is a copy of the adj_matrix of the original graph
    adj_obs = adj_matrix.copy()
    # number of edges in target set
    pos_num = len(test_edges)
    # number of non_edges in target set
    neg_num = len(test_non_edges)

    # test_set is the union of test edges and non-edges
    test_set = np.concatenate((test_edges, test_non_edges), axis=0)

    # --remove edges by chaning adj_obs
    for edge in test_edges:
        u, v = edge
        adj_obs[u, v] = 0
        adj_obs[v, u] = 0

    return adj_obs, test_set


# compute Jaccard similarity for all node pairs
def compute_Jaccard(adj, node_pairs):  # cn/d_u + d_v - cn
    # adj is the adjacency matrix of the observed network
    pair_num = node_pairs.shape[0]
    # store the similarity scores for all the node pairs
    sim_vec = np.zeros(pair_num, dtype=float)

    for i in range(pair_num):
        u = node_pairs[i, 0]
        v = node_pairs[i, 1]

        # number of common neighbors
        nbrs_u = set(np.where(adj[u, :] == 1)[0])
        nbrs_v = set(np.where(adj[v, :] == 1)[0])
        cn_num = cn_num = len(nbrs_u & nbrs_v)
        # degree of u and v
        deg_u = len(nbrs_u)
        deg_v = len(nbrs_v)
        # similarity between u and v
        sim_vec[i] = cn_num / (deg_u + deg_v - cn_num)

    return sim_vec


# compute Katz similarity for all node pairs
def compute_Katz(adj, node_pairs):
    # adj is the adjacency matrix of the observed network
    pair_num = node_pairs.shape[0]
    sim_vec = np.zeros(pair_num, dtype=float)

    # compute similarity matrix
    n = adj.shape[0]
    I = np.eye(n)
    beta = 0.09
    Katz_matrix = np.linalg.inv(I - beta * adj) - I

    # get the Katz similarity for each pair of node
    for i in range(pair_num):
        u = node_pairs[i, 0]
        v = node_pairs[i, 1]
        sim_vec[i] = Katz_matrix[u, v]
    return sim_vec


# link prediction, by comparing similarity with a threshold theta
def link_pred(adj, node_pairs, metric, theta):
    # adj is the adjacency matrix of the observed network
    # node_pairs are all the pairs of node
    # metric = 'Jaccard' or 'Katz'
    # theta is the threshold

    # first, compute the similarity
    if metric == "Jaccard":
        sim_vec = compute_Jaccard(adj, node_pairs)
    elif metric == "Katz":
        sim_vec = compute_Katz(adj, node_pairs)
        # print(sim_vec)

    # make prediction

    pred = np.zeros(len(node_pairs), dtype=int)

    # if sim_vec[i] >= theta, predict node pair i as a link (set pred[i] as 1)
    # otherwise set pred[i] as 0
    for i in range(len(sim_vec)):
        if sim_vec[i] >= theta:
            pred[i] = 1  # Predict a link
        else:
            pred[i] = 0  # Predict no link

    return pred
