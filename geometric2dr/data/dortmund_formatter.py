"""
Gexifier for TU Dortmund graph kernel based datasets.

This class helps turn datasets from the format with which TU Graph Kernel datasets 
are written into Gexf datasets, which Geo2DR can work with

It reads the DS_A.txt, DS_graph_indicator.txt, and DS_graph_labels.txt to create a 
folder of graphs in GEXF format and a graph-id to graph-classification label file.

The saved format will be 
dataset_name/dataset_name/<name>.gexf : folder containing individual gexf files of each graph.
dataset_name/dataset_name.Labels : a file denoting each gexf file to the integer class label
"""


import pickle
import sys
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


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


class DortmundGexf(object):
	"""
	A class which reads TU Dortmund style datasets and processes them
	into a corresponding set of .gexf graphs and an associated .Labels file

	See tu_gexifier for a more basic script based version. This class version
	will also contain various metadata about the dataset which may be useful
	for downstream decomposition algorithms and other analysis
	"""
	def __init__(self, dataset, path_to_dataset, output_dir_for_graph_files):
		super(DortmundGexf, self).__init__()
		self.dataset = dataset
		# TODO Do some regex dark magicks 
		self.graph_A_fname = path_to_dataset + dataset + "/" + dataset + "_A.txt"
		self.graph_indicator_fname = path_to_dataset + dataset + "/" + dataset + "_graph_indicator.txt"
		self.graph_labels_fname = path_to_dataset + dataset + "/" + dataset + "_graph_labels.txt"
		self.node_labels_fname = path_to_dataset + dataset + "/" + dataset + "_node_labels.txt"
		self.output_dir_for_graph_files = output_dir_for_graph_files
		self.folder_for_graph_files = output_dir_for_graph_files + dataset

	def format_dataset(self):  
		if os.path.isdir(self.folder_for_graph_files):
			print("#... The dataset %s already exists, closing program ...#" % (self.folder_for_graph_files))
		else:
			print("#... Starting gexification ...#")
			# Read each line in the graph_indicator and add a node_id (of the dataset) to its corresponding 
			# graph indicated by the value
			print("#... Generating graph nodes dictionary ...#")
			graph_nodes = defaultdict(list) # each graph gets its list of nodes
			nodes = open(self.graph_indicator_fname).readlines()
			nodes = [int(x.strip()) for x in nodes]

			nodes_to_graph = {}

			node_id = 1
			for gindex in nodes:
				graph_nodes[gindex].append(node_id)
				nodes_to_graph[node_id] = gindex
				node_id += 1
			del(nodes)

			# Now get the edges for each graph making sure the edge nodes are actually in the graph as well
			print("#... Generating graph edges dictionary ...#") # Choke point of algorithm (but we realistically only do it once.)
			graph_edges = defaultdict(list)
			edges = open(self.graph_A_fname).readlines()
			edges = [x.split(",") for x in edges]
			edges = [(int(x.strip()), int(y.strip())) for x,y in edges] # nice little list of tuples

			todoedges = len(edges)
			# for x,y in tqdm(edges):
			# 	## TAKES A LOT OF TIME, TODO: parallel or smarter algo (it gets worse over time)
			# 	for gindex in graph_nodes.keys():
			# 		if (x in graph_nodes[gindex] and y in graph_nodes[gindex]):
			# 			graph_edges[gindex].append((x,y))
			# 			break # no need to continue going through graphs checking for this edge
			# 		else:
			# 			continue

			# new version
			new_graph_edges = defaultdict(list)
			for x,y in tqdm(edges):
				x_gindex = nodes_to_graph[x]
				y_gindex = nodes_to_graph[y]
				if x_gindex == y_gindex:
					new_graph_edges[x_gindex].append((x,y))
			del(edges)

			# self.graph_edges = graph_edges
			# self.new_graph_edges = new_graph_edges
			# Check tested.

			print("#... Generating NX Graph dictionary ...#")
			# The more you know, the defaultdict is a factory pattern
			graph_nx = defaultdict()
			for gindex in tqdm(graph_edges.keys()):
				G = nx.Graph()
				for u,v in graph_edges[gindex]:
					G.add_edge(u,v)
				graph_nx[gindex] = G
			del(graph_edges)

			print("#... Relabeling nodes as necessary via attribute file ...#")
			# Our system finds unique substructures across the dataset using the node labels
			# If the nodes have labels they will be found in the node_labels_fname
			# Else if we are dealing with unlabelled graphs we will use the degree of the nodes as Labels
			if os.path.isfile(self.node_labels_fname):
				## This also takes time
				print("#... Relabeling nodes using %s ...#" % (self.node_labels_fname))
				node_att_relabel = {}
				new_node_labels = open(self.node_labels_fname).readlines()
				old_node_label = 1
				for new_node_label in new_node_labels:
					node_att_relabel[old_node_label] = new_node_label.strip()
					old_node_label += 1

				for gindex in graph_nx.keys():
					if gindex % 1000 == 0:
						print("Setting node label att %s" % (str(gindex)))
					nx.set_node_attributes(graph_nx[gindex], node_att_relabel, 'Label')

			else:
				print("#... Could not find node labeling file, will label by degree ...#")
				for gindex in graph_nx.keys():
					node_att_relabel = {}
					G = graph_nx[gindex]
					for node in G.nodes:
						if gindex % 50 == 0:
							print("Setting node label att %s" % (str(gindex)))
						node_att_relabel[node] = G.degree(node)
					nx.set_node_attributes(graph_nx[gindex], node_att_relabel, 'Label')

			print ("#... Physical relabeling of nodes as necessary ...#")
			for gindex in tqdm(sorted(graph_nx.keys())):
				graph_nx[gindex] = nx.convert_node_labels_to_integers(graph_nx[gindex], first_label=1)

			print ("#... Generating graph classification labels ...#")
			graph_classes = defaultdict()
			labels = open(self.graph_labels_fname).readlines()
			labels = [x.strip() for x in labels]

			graph_id = 1
			for label in labels:
				graph_classes[graph_id] = label
				graph_id += 1

			print("#... Generating graph dataset dictionaries completed ...#")

			#################################################
			### Writing to files in and making the labels ###
			#################################################
			print("#... Writing to gexf files ...#")
			# We need a gexf file for each graph and a folder to put them into
			if os.path.isfile(self.folder_for_graph_files):
				print("Already have the folder %s" % (self.folder_for_graph_files))
				for gindex in sorted(graph_nx.keys()):
					graph_fname = str(gindex)+ ".gexf"
					graph_gexf_path = self.folder_for_graph_files + "/" + graph_fname
					nx.write_gexf(graph_nx[gindex], graph_gexf_path)

			else:
				print ("Dont have folder %s, creating and writing gexf files there" % (self.folder_for_graph_files))
				os.makedirs(self.folder_for_graph_files)
				for gindex in sorted(graph_nx.keys()):
					graph_fname = str(gindex) + ".gexf"
					graph_gexf_path = self.folder_for_graph_files + "/" + graph_fname
					nx.write_gexf(graph_nx[gindex], graph_gexf_path)

				# Write the labels
				print("Writing classification labels file for graphs in dataset")
				name_of_labels_file = self.output_dir_for_graph_files + self.dataset + ".Labels"
				with open(name_of_labels_file, "w") as fh:
					for gindex in sorted(graph_classes.keys()):
						graph_fname = str(gindex) + ".gexf"
						fh.write("%s %s\n" % (graph_fname, graph_classes[gindex]))
					

if __name__ == '__main__':
	gexifier = DortmundGexf("MUTAG", "dortmund_data/", "/tmp/")
	gexifier.format_dataset()