from graphLib import Graph
from scipy.sparse import linalg
from scipy import sparse
import scipy.sparse as ssp
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import random

# =================== Task 4: Generate Random Graphs ===================

# parameters - do not change the seed
seed = 14
n = 200
p = 0.2
m = 5


# generate ER graph
def gen_ER(n, p, seed):
    # n: number of nodes
    # p: probability

    # dont change this line
    np.random.seed(seed)

    # number of all possible edges
    edge_num = int(n * (n - 1) / 2)

    # generate a random vector
    # one random number for each possible edge
    edge_roll = np.random.uniform(0.0, 1.0, edge_num)

    # initialize the adj matrix
    A = np.zeros((n, n), dtype=int)

    # pointer to the current edge
    current = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # compare the random number edge_roll[current] with p
            # to decide if we add this edge
            if edge_roll[current] < p:
                # remember to update two entries in A
                A[i][j] = 1
                A[j][i] = 1

            # move to the next edge
            current += 1
    return A


# an auxiliary function to generate BA graph
# select m unique nodes from input nodes
def pick_nodes(nodes, m):

    # initialize the target nodes as an empty set
    target_nodes = set()

    while len(target_nodes) < m:  # keep adding nodes until m nodes are added
        # randomly select a node
        node = random.choice(nodes)

        # use add() function to add node into target_nodes
        # note: set object stores unique elements;
        # for example, if target_nodes = {1,2,3}, when add a node 1 into target_nodes, it will not change

        target_nodes.add(node)

    return target_nodes


# generate BA graph
def gen_BA(n, m, seed):

    # --- initialize adj matrix
    A = np.zeros((n, n), dtype=int)

    # target nodes for new edges
    # initially, the target_nodes are 0,1,2,...,m-1
    target_nodes = list(range(m))

    # list of existing nodes; node will repeat once for each adjacent edge
    repeated_nodes = []

    # add a new node to graph;
    # initially, the first new node is m
    new_node = m

    # don't change this line
    random.seed(seed)

    # add node m,m+1,m+2,...,n to the graph
    while new_node < n:

        # add edges between new_node the each node in target_nodes
        edges = zip([new_node] * m, target_nodes)
        for edge in edges:
            A[edge[0], edge[1]] = 1
            A[edge[1], edge[0]] = 1

        # --- now,  generate the new target_nodes, according to the node degrees

        # add nodes in target_nodes into repeated nodes
        repeated_nodes.extend(target_nodes)

        # add new_nodes m times into repeated nodes
        repeated_nodes.extend([new_node] * m)

        # --- pick m nodes from repeated_nodes, which are the new target_nodes

        target_nodes = pick_nodes(repeated_nodes, m)

        # --- update new_node
        new_node += 1

    return A
