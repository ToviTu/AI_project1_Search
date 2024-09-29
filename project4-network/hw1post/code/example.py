# to use Graph, import it from graphLib
from graphLib import Graph
# process sparse matrix
import scipy.sparse as ssp


# an example graph is provided in '../data/example_graph.npz'

# read the file and construct a graph 
graph = Graph(ssp.load_npz('../data/example_graph.npz'))


# number of nodes in graph:
n = graph.n
print(n)

# get the adjacency matrix of graph

A = graph.get_adj_matrix()
print(A)

# get adjacency list

adj_list = graph.get_adj_list()
for i in range(n):
	print(i, adj_list[i])

# get degree sequence

deg_seq = graph.get_deg_seq()
print(deg_seq)


