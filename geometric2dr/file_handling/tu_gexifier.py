"""
Gexifier for TU kernel based datasets.

This scrips helps turn datasets from the format with which TU Graph Kernel datasets are written into something we can work with.

It reads the DS_A.txt, DS_graph_indicator.txt, and DS_graph_labels.txt to create a folder of graphs in GEXF format and a graph-id to graph-classification label file.

The saved format will be 
dataset_name/dataset_name : folder containing individual gexf files of each graph.
dataset_name/dataset_name.Labels : a file denoting each gexf file to the integer class label
"""

import pickle
import sys
import os
import networkx as nx
import argparse

from collections import defaultdict

parser = argparse.ArgumentParser(description='Tool to process TU graph kernel datasets')
parser.add_argument("-d", "--dataset", help="Path of the directory containing the dataset in TU format")

args = parser.parse_args()

################################################
############# Settings and Files ###############
################################################

dataset = args.dataset
graph_A_fname = "dortmund_data/" + dataset + "/" + dataset + "_A.txt"
graph_indicator_fname = "dortmund_data/" + dataset + "/" + dataset + "_graph_indicator.txt"
graph_labels_fname = "dortmund_data/" + dataset + "/" + dataset + "_graph_labels.txt"
node_labels_fname = "dortmund_data/" + dataset + "/" + dataset + "_node_labels.txt"

folder_for_graph_files = "dortmund_gexf/" + dataset

################################################
################################################
# Explanation of the TU graph kernel files

# n = total number of nodes
# m = total number of edges
# N = number of graphs

# DS_A.txt (m lines):	represents a sparse (block diagonal) adjacency matrix for all graphs, 
#						each line corresponds to (row, col) position resp. (node_id, node_id).
#						All graphs are undirected. Hence, DS_A.txt contains two entries for 
#						each edge.

# DS_graph_indicator.txt (n_lines): column vector of graph identifiers for all nodes of all graphs
#									the value in the i-th line is the graph_id of the node with 
#									node_id i

# DS_graph_labels.txt (N lines):	class labels for all graphs in the dataset, the value in the i-th
#									line is the class label of the graph with graph_id i

# DS_node_labels.txt (n lines):		column vector of node labels, the value in the i-th line is the
#									node label for node i

# There are optional files if the respective information is available:
# DS_edge_labels.txt (m lines): same size as DS_A.txt: labels for the edges in DS_A.txt
# DS_edge_attributes.txt (m lines): same size as DS_A.txt, attributes for the edges in DS_A.txt
# DS_node_attributes.txt (n lines): matrix of node attributes, the comma seperated values in the i-th line is the feature vector of the node with node_id i
# DS_graph_attributes.txt (N lines): regression values for all graphs in the datset, the value in the i-th line is the attribute of the graph with graph_id i

#################################################
##### Generating Dictionaries from files ########
#################################################