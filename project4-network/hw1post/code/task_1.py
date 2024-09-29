import numpy as np
import scipy.sparse as ssp
from graphLib import Graph
from itertools import combinations
import matplotlib.pyplot as plt

#=================== Task 1: Compute Graph Properties ===================

#------------- Task 1.1: compute degree distribution
def compute_deg_distr(graph):
    # compute the degree distribution of the input graph
    # return two array's
    # degs: all the possible node degrees, in ascending order
    # counts: the corresponding counts (number of nodes) for each degree value in degs
   
    # get the degree sequence of all the nodes
    

    # get degree and its corresponding count
    # using np.unique(); check documentation
    #degs, counts = np.unique()
    degs = []
    counts = []

    # The adj list is handy for this task
    all_degrees = []
    adj_list = graph.get_adj_list()
    for node in range(graph.n):
        deg_node = adj_list[node]
        all_degrees.append(len(deg_node))
    
    degs, counts = np.unique(all_degrees, return_counts=True)
    
    return degs, counts

#------------ Task 1.2: compute clustering coefficient
def get_cc_local(graph,nid):
    
    # get the neighbors of node nid
    adj_list = graph.get_adj_list()
    nbrs = adj_list[nid]
    
    # number of neighbors of node nid 
    deg = len(nbrs)

    # get adjacency matrix of graph in dense form
    adj_matrix = graph.get_adj_matrix()

    # all possible pairs of neighbors of nid
    all_pairs = deg * (deg - 1) / 2
    
    # number of connected pairs of neighbors
    closed_pairs = 0
        
    if deg < 2:
        return 0.0
    else:
        # enumerate all pairs of neighbors
        comb = combinations(nbrs, 2) 

        for node_pair in list(comb):              
            # check if this pair of nodes are connected and update closed_pairs
            if adj_matrix[node_pair[0], node_pair[1]] == 1:
                closed_pairs += 1

        # return local clustering coefficient
        return closed_pairs / all_pairs

def get_cc_global(graph):
    adj_matrix = graph.get_adj_matrix()
    
    # number of closed 2-path
    closed_2_path = 0
    
    # number of 2-path 
    all_2_path = 1
    
    #loop over all nodes
    for i in range(graph.n):
        # get the neighbors of node i
        nbrs = graph.get_adj_list()[i]
        nbrs_num = len(nbrs)
        if nbrs_num >= 2:
            # enumerate all pairs of neighbors
            comb = combinations(nbrs, 2)      
            for node_pair in list(comb):
                # update all_2_path
                all_2_path += 1

                #update closed_2_path
                if adj_matrix[node_pair[0], node_pair[1]] == 1:
                    closed_2_path += 1

    # return global clustering coefficient
    return closed_2_path / all_2_path

# --------------------- Task 1.3: compute diameter
def distance_BFS(adj_list, s):
    # return the distance of s to every node
    # travel the graph from s using BFS; this is will create a tree rooted at s
    # the hight of the tree is the longest distance

    n = len(adj_list)

    # use a vector to record if a node is visited or not
    visited = [False] * n   
    visited[s] = True
    
    # store the distance from s to all other nodes
    distance = np.zeros(n, dtype = int)
    distance[s] = 0
    
    # current layer of nodes
    current_level = [s]
    # next layer of nodes
    next_level = [] # The neighbors of node s

    # the number of layers
    depth = 0
    
    # while current layer is not empty
    while len(current_level)>0:
        for node in current_level: # each node in current layer
            nbrs = adj_list[node]
            for nbr in nbrs: #for each neighbor of this node
                # add a neighbor into next layer, if the neighbor is not visited yet
                if not visited[nbr]:
                    visited[nbr] = True
                    next_level.append(nbr)
                
                # remember to update visited


        # update depth
        depth += 1
        # update distance from s to all nodes in the next_level
        for child in next_level:
            distance[child] = depth


        # set current_level as next_level
        current_level = next_level
        # empty next_level
        next_level = []

    return distance

def get_diameter(graph):
    adj_list = graph.get_adj_list()

    diameter = -1

    # treat every node in graph as the source node
    # call distance_BFS() to get the distance
    # find the max distance
    for source in range(graph.n):
        distance = distance_BFS(adj_list, source)
        max_distance = np.max(distance)
        if max_distance > diameter:
            diameter = max_distance


    return diameter
