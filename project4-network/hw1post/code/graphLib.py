import numpy as np
import scipy.sparse as ssp
from collections import defaultdict 

# by default, we will store and read graph from sparse adjcency matrix

class Graph:

    def __init__(self, graph_adj_matrix = None):
        #graph_adj_matrix is sparse matrix
        if graph_adj_matrix is not None:
            self.adj_sparse = graph_adj_matrix
            self.n = graph_adj_matrix.shape[0]

    def get_adj_matrix(self):
        # get the adj in dense form (ndarray)
        return self.adj_sparse.toarray()

    def get_edge_list(self):
        # get the edge list of the graph
        # [..., (i,j), ...], requirement i < j
        rows,cols = self.adj_sparse.nonzero()
        edge_list = []
        for row, col in zip(rows,cols):
            if row < col:
                edge_list.append([row,col])
        return edge_list

    def get_adj_list(self):
        # get the adjacency list of graph
        # return a dictionary
        adj_matrix = self.get_adj_matrix()
        adj_dict = defaultdict(list)
        for u in range(self.n):
            #get the neighbors of node u
            nbrs_u = np.nonzero(adj_matrix[u,:])[0]
            for nbr in nbrs_u:
                adj_dict[u].append(nbr)
        return adj_dict

    def get_deg_seq(self):
        return sum(self.get_adj_matrix())