"""
Reverse gexifier for data format used in Geo2DR to 
the format in Yanardag and Vishwanathan Deep Graph Kernels

Author: Paul Scherer

This script will have to be run in python 2 for sake of
matching the pickling procedure
"""


# Data description from README
# === DATASETS === 
# Please visit www.mit.edu/~pinary/kdd address to download the datasets 
# used in the paper, and extract them under "datasets" folder.

# Each dataset is a dictionary saved as a pickle. The dictionary of the dataset 
# looks as follows:

# dataset = {"graph": graphs, "labels": labels}

# labels: this is an array where each element corresponds to the class of an individual graph. The class 
# labels are expected to be integers.

# graphs: this is a dictionary where each key represents the index number for a graph. so if the dataset has 
# 600 graphs, then this is a dictionary that has 600 keys, ranged from 0 to 599. 
# Each graph in this dictionary points to another dictonary that represents all the nodes embedded in the graph, 
# where nodes themselves also have index numbers 0 to max_number_of_nodes. For instance, accessing to a node 
# indexed with "nidx" in the graph indexed as "gidx" reveals the following dictionary:

# graph_data[gidx][nidx] = {"neighbors": list_of_neighbors, "label": label_of_the_node}

# where list_of_neighbors is an array that represents all other node id's that node "nidx" 
# has an edge, and label_of_the_node represents the label associated with the node. labels 
# are not used in graphlet kernel since it works on unlabeled graphs.

# As an example, here is how the dictionaries look for mutag dataset (can be found under 
# datasets/mutag.graph).

# # load the data
# >>> import pickle
# >>> f = open("datasets/mutag.graph")
# >>> data = pickle.loads(f.read())
# >>> f.close()

# # the fields this data has
# >>> data.keys()
# ['graph', 'labels']

# # let's look at the labels array
# >>> data['labels']
# array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
#        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
#        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
#        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
#        -1], dtype=int16)

# # these are the graph indices (mutag has 188 graphs)
# >>> data['graph'].keys()
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187]

# # let's access to first graph (indexed by 0) and its first node (indexed by 0)
# # here, [1, 13] are two nodes that node 0 is connected to, and 'label' represents 
# # the discrete label associated with label 3. 
# >>> data['graph'][0][0]
# {'neighbors': array([ 1, 13], dtype=uint8), 'label': (3,)}

import os
from collections import Counter, defaultdict
import numpy as np
import pickle

import networkx as nx
import geometric2dr.embedding_methods.utils as utils


def load_graph(file_handle):
	"""
	Load the gexf format file as a nx_graph and an adjacency matrix.
	"""
	graph = nx.read_gexf(file_handle)
	adj_matrix = nx.to_numpy_matrix(graph)
	return graph, adj_matrix


dataset = "MUTAG"
path_to_gexf_data = "data/"
graph_class_labels_fh = path_to_gexf_data+ dataset + ".Labels"
dataset_path = path_to_gexf_data + dataset

# Yanardag style dataset 
data = {}
labels = []
graph_files = {}

graph_files = utils.get_files(dataset_path, extension=".gexf", max_files = 0)
label_tuples = utils.get_class_labels_tuples(graph_files, graph_class_labels_fh)
graph_classes = np.array([y for z,y in sorted(label_tuples, key=lambda x: x[0])])
data['labels'] = graph_classes
gf = graph_files[0]

graph_data = {}

for gf in graph_files:
	gindex = int(os.path.basename(gf).split(".")[0])-1
	nx_graph, adj_matrix = load_graph(gf)


	graph_data[gindex] = {}
	for node_string in nx_graph.nodes():
		node_label = int(nx_graph.nodes[node_string]['Label'])
		node_id = int(node_string)

		neighbors = list(nx_graph.neighbors(node_string))
		neighbors = [int(x) for x in neighbors]

		graph_data[gindex][node_id] = {}
		graph_data[gindex][node_id]['neighbors'] = np.array(neighbors, dtype="uint8")
		graph_data[gindex][node_id]['label'] = node_label

data['graph'] = graph_data


with open('mutag' + ".graph", 'wb') as handle:
	pickle.dump(data, handle, protocol=2)