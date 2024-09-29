from graphLib import Graph
from scipy.sparse import linalg
import scipy as sp
import scipy.sparse as ssp
import numpy as np
from collections import defaultdict
from itertools import combinations

# =================== Task 1: Compute Node Centralities ===================


# ---------- Task 2.1: eigenvector centrality
def get_eigen_centrality(graph):
    # adj matrix of graph
    A = graph.get_adj_matrix()
    # change element in A into float type
    A_float = A.astype(float)

    # use linalg.eigs() to get eigenvalue and vectors
    eigenvalue, eigenvector = linalg.eigs(A_float, k=1, which="LR")

    largest = eigenvector.flatten().real

    # compute the norm
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)

    # return the normalized eigen vector
    return largest / norm


# ------------ Task 2.2:  compute betweenness centrality
# compute the levels for a given source node s
def levels_BFS(graph, s):
    # this function is essentially the same as distance_BFS()
    # the only difference is that we need to store all the levels of node

    adj_list = graph.get_adj_list()
    visited = [False] * graph.n

    # store all levels of nodes
    levels = []

    current_level = [s]
    visited[s] = True

    next_level = []
    depth = 0
    distance = np.zeros(graph.n, dtype=int)
    distance[s] = 0

    while len(current_level) > 0:
        for node in current_level:
            for nbr in adj_list[node]:
                # comment out next line once loop is implemented
                if not visited[nbr]:
                    visited[nbr] = True
                    next_level.append(nbr)
                # print(nbr)

        # update depth
        depth += 1
        # update distance for all the node in the next_level
        for i in next_level:
            # comment out next line once loop is implemented
            distance[i] = depth
            # print(i)

        # add the current level of nodes into levels
        levels.append(current_level)

        # update current_level and next_level
        current_level = next_level

        next_level = []

    return distance, levels


# compute predecessors for all nodes other than s
def find_pred(graph, levels, s):

    adj_list = graph.get_adj_list()

    # initialize a dictionary
    # key is the node id, value is a list of neighbors
    pred_dict = defaultdict(list)
    depth = len(levels)

    # iterate from the second level to the last level
    for level_num in range(1, depth):
        # nodes in the current level
        current_level = levels[level_num]
        # nodes in the previous level
        prev_level = levels[level_num - 1]

        # find the predecessors for each node in current level
        for node in current_level:

            # predecessors are the intersection of neighbors and prev_level
            nbrs = list(set(adj_list[node]) & set(prev_level))
            pred_dict[node] = nbrs

    return pred_dict


# find number of shortest paths from source node s to all other nodes
def find_num_shortest_path(graph, levels, pred_dict, s):
    # store the number of shortest paths
    num_shortest_path = np.zeros(graph.n, dtype=int)
    num_shortest_path[s] = 1  # by default

    depth = len(levels)
    # iterate from the second level to the last level
    for level_num in range(1, depth):
        current_level = levels[level_num]
        # calculate number of shortest paths from s to every node in the current level
        for node in current_level:
            # get the predecessors of node
            pred = pred_dict[node]
            for p_node in pred:
                num_shortest_path[node] += num_shortest_path[p_node]

    return num_shortest_path


# fill in the two matrices
def build_matrix(graph, distance_mat, num_shortest_path_mat):
    # treat each node in graph as the source node
    for s in range(graph.n):
        # get the distance and levels
        distance, levels = levels_BFS(graph, s)

        # fill in the s-th row of distance_mat matrix
        distance_mat[s, :] = distance

        # find the predecessors for all nodes
        pred_dict = find_pred(graph, levels, s)

        # compute the number of shortest paths from s to all other nodes
        num_shortest_path = find_num_shortest_path(graph, levels, pred_dict, s)

        # fill in the s-th row of num_shortest_path_mat matrix
        num_shortest_path_mat[s, :] = num_shortest_path


# compute betweenness centrality for a node w, using the two matrices
def get_btw_c(graph, w, distance_mat, num_shortest_path_mat):
    # all other nodes other than w
    rest_nodes = list(range(graph.n))
    rest_nodes.remove(w)

    # initialize centrality as 0
    btw_c = 0.0

    # enumerate each pair of nodes
    comb = combinations(rest_nodes, 2)
    for node_pair in list(comb):
        u = node_pair[0]
        v = node_pair[1]

        # get the distances for the three pairs of nodes (u,v), (w,v), (w,u)
        d_uv = distance_mat[u, v]
        d_wv = distance_mat[w, v]
        d_wu = distance_mat[w, u]
        if d_uv == d_wv + d_wu:  # check if w is on the shortest path of u and v
            # compute number of shortest path (nsp) for three pairs of nodes (u,v), (w,v), (w,u)
            nsp_uv = num_shortest_path_mat[u, v]
            nsp_wu = num_shortest_path_mat[w, u]
            nsp_wv = num_shortest_path_mat[w, v]
            # update btw_c
            btw_c += nsp_wu * nsp_wv / nsp_uv

    # return btw_c;
    # *** remember to normalize ***
    return 2 * btw_c / (graph.n - 1) / (graph.n - 2)
