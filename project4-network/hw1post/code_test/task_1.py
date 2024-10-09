import numpy as np
import scipy.sparse as ssp
from graphLib import Graph
from itertools import combinations
import matplotlib.pyplot as plt

# =================== Task 1: Compute Graph Properties ===================


# ------------- Task 1.1: compute degree distribution
def compute_deg_distr(graph):
    # compute the degree distribution of the input graph
    # return two array's
    # degs: all the possible node degrees, in ascending order
    # counts: the corresponding counts (number of nodes) for each degree value in degs

    # get the degree sequence of all the nodes

    # get degree and its corresponding count
    # using np.unique(); check documentation
    # degs, counts = np.unique()
    degs = []
    counts = []

    deg_seq = graph.get_deg_seq()
    degs, counts = np.unique(deg_seq, return_counts=True)

    return degs, counts


# ------------ Task 1.2: compute clustering coefficient
def get_cc_local(graph, nid):

    # get the neighbors of node nid
    nbrs = graph.get_adj_list()[nid]

    # number of neighbors of node nid
    deg = len(nbrs)

    # get adjacency matrix of graph in dense form
    adj_matrix = graph.get_adj_matrix()

    # all possible pairs of neighbors of nid
    all_pairs = combinations(nbrs, 2)

    # number of connected pairs of neighbors
    closed_pairs = 0

    if deg < 2:
        return 0.0
    else:
        # enumerate all pairs of neighbors
        comb = combinations(nbrs, 2)

        for node_pair in list(comb):
            # check if this pair of nodes are connected and update closed_pairs
            u, v = node_pair
            closed_pairs += 1 if adj_matrix[u, v] == 1 or adj_matrix[v, u] == 1 else 0

        # return local clustering coefficient
        return (2 * closed_pairs) / (deg * (deg - 1))


def get_cc_global(graph):
    # Get adjacency matrix of the graph
    adj_matrix = graph.get_adj_matrix()

    closed_2_path = 0  # Number of closed 2-paths (triangles)
    all_2_path = 0  # Number of 2-paths (open and closed triplets)

    # Loop over all nodes
    for i in range(graph.n):
        # Get the neighbors of node i
        nbrs = np.where(adj_matrix[i] == 1)[0]  # Neighbors of node i
        nbrs_num = len(nbrs)  # Number of neighbors

        if nbrs_num >= 2:
            # Enumerate all pairs of neighbors
            comb = combinations(nbrs, 2)
            for node_pair in comb:
                all_2_path += 1  # Every pair of neighbors forms a 2-path

                # Check if the two neighbors are connected (i.e., form a triangle)
                if adj_matrix[node_pair[0], node_pair[1]] == 1:
                    closed_2_path += 1  # Found a closed 2-path (triangle)

    # Return the global clustering coefficient
    if all_2_path == 0:
        return 0  # Avoid division by zero
    return closed_2_path / all_2_path


# --------------------- Task 1.3: compute diameter
# def distance_BFS(adj_list, s):
#     # return the distance of s to every node
#     # travel the graph from s using BFS; this is will create a tree rooted at s
#     # the hight of the tree is the longest distance

#     n = len(adj_list)

#     # use a vector to record if a node is visited or not
#     visited = [False] * n
#     visited[s] = True

#     # store the distance from s to all other nodes
#     distance = np.zeros(n, dtype=int)
#     distance[s] = 0

#     # current layer of nodes
#     current_level = [s]
#     # next layer of nodes
#     next_level = []

#     # the number of layers
#     depth = 0

#     # while current layer is not empty
#     while len(current_level) > 0:
#         for node in current_level:  # each node in current layer
#             nbrs = []
#             for nbr in nbrs:  # for each neighbor of this node
#                 # add a neighbor into next layer, if the neighbor is not visited yet
#                 nbr = 0

#                 # remember to update visited

#         # update depth
#         depth = 0
#         # update distance from s to all nodes in the next_level
#         for child in next_level:
#             distance[child] = 0

#         # set current_level as next_level
#         current_level = []
#         # empty next_level
#         next_level = []

#     return distance

from collections import deque


def distance_BFS(adj_list, s):
    # Get the number of nodes
    n = len(adj_list)

    # Vector to record if a node is visited or not
    visited = [False] * n
    visited[s] = True

    # Store the distance from s to all other nodes, initialized to -1 for unreachable nodes
    distance = np.full(n, -1, dtype=int)
    distance[s] = 0

    # Queue for BFS traversal
    queue = deque([s])

    # Perform BFS
    while queue:
        # Dequeue a node from the current layer
        node = queue.popleft()

        # Get the neighbors of this node
        nbrs = adj_list[node]

        # For each neighbor, if not visited, mark it as visited and update distance
        for nbr in nbrs:
            if not visited[nbr]:
                visited[nbr] = True
                distance[nbr] = distance[node] + 1
                queue.append(nbr)

    return distance


def get_diameter(graph):
    adj_list = graph.get_adj_list()

    diameter = -1

    # treat every node in graph as the source node
    # call distance_BFS() to get the distance
    # find the max distance
    for source in range(graph.n):
        distances = distance_BFS(adj_list, source)
        max_distance = np.max(distances)

        if max_distance > diameter:
            diameter = max_distance

    return diameter
