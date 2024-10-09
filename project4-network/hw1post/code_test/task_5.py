import numpy as np
import scipy.sparse as ssp
from graphLib import Graph
import matplotlib.pyplot as plt
from scipy import sparse

# =================== Task 5: Inuence Maximization and Blocking ===================

# parameters
# p = 0.05
# mc = 5000
# k = 10


# ------------------------- Task 5.1: influence maximization
# simulate a single diffusion process
def info_diff(graph, seed_set, p):
    # seed_set: the initial set of seeds
    # p: universal activation probability
    # output: the number of activated nodes
    adj_list = graph.get_adj_list()

    n = graph.n
    adj_list = graph.get_adj_list()

    if len(seed_set) == 0:
        return 0
    # store the active_nodes
    active_nodes = list(seed_set)

    # store currently active nodes
    current_active = list(seed_set)

    # active nodes are flagged as 1
    activated_flag = np.zeros(n, dtype=int)

    activated_flag[seed_set] = 1

    while len(current_active) > 0:
        # use the nodes in current_active to activate others
        newly_activated = []
        for (
            u
        ) in (
            current_active
        ):  # each node in currect_active has the chance to activate neighbors
            # neighbors of nbrs
            nbrs = adj_list[u]
            for v in nbrs:  # add v into newly_activated according to the model
                if activated_flag[v] == 0:
                    # Activate v with probability p
                    rnd = np.random.uniform(0.0, 1.0)
                    if rnd < p:
                        newly_activated.append(v)
                        activated_flag[v] = 1  # Mark v as activated

        # add newly activated nodes into active nodes
        active_nodes.extend(newly_activated)

        # update current active nodes
        current_active = newly_activated

    return len(active_nodes)


# estimate the influece of a set of seeds
def get_influence(graph, seed_set, p, mc):
    # seed_set: a given set of seeds
    # mc: simulation times

    np.random.seed(14)

    # initialize the total number of active nodes throughout simulations
    total_active_num = 0

    # Perform mc simulations
    for i in range(mc):
        # Run a single diffusion simulation and update total_active_num
        total_active_num += info_diff(graph, seed_set, p)

    return total_active_num / mc


import tqdm


# find the optimal set of seeds using greedy search
def greedySearch(graph, k, p, mc):
    n = graph.n
    # find seeds in a candidate set
    # initially, candidate set contains all the nodes
    candidate = list(range(n))
    seed_set = []  # store the seeds

    influence = [0.0]  # store the influece of each set of seeds
    for i in tqdm.tqdm(range(k)):  # repeat k times to find k seeds

        # influence of current set of seeds
        current_influence = get_influence(graph, seed_set, p, mc)

        # --- find most influential node in the candidate set
        # influence of a node = influence of ( current_set + node) -  influence of current_set
        max_gain = -1
        most_influence_node = -1

        for u in candidate:
            new_seed_set = seed_set + [u]
            new_influence = get_influence(graph, new_seed_set, p, mc)
            gain = new_influence - current_influence

            if gain > max_gain:
                max_gain = gain
                most_influence_node = u

        # most_influence_node is the node with the highest influence
        # add it to the seed_set
        seed_set.append(most_influence_node)

        # store the influence of current set of seeds
        influence.append(current_influence + max_gain)

        # remove most_influence_node from candidate set
        candidate = np.setdiff1d(candidate, most_influence_node)

    return seed_set, influence


# ----------------- Task 5.2: random blocking ------------
# modify the graph by deleting edges
def modify_graph(graph, target_nodes):
    adj_list = graph.get_adj_list()
    A = graph.get_adj_matrix()

    np.random.seed(13)

    # construct the adj matrix of a new graph
    A_new = A.copy()

    for u in target_nodes:
        nbrs = adj_list[u]
        # randomly select a neighbor
        v = np.random.choice(nbrs, 1)
        A_new[u, v] = 0
        A_new[v, u] = 0

        # delete the edge; remember to modify two entries

    # transfer to scipy.sparse.csr.csr_matrix
    return sparse.csr_matrix(A_new)
