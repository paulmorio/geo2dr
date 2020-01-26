"""
Shortest path based decomposition algorithm to create graph documents 

The main use case for this script is for the user to supply a path to the directory
containing the .gexf graph files of a dataset. The decomposition function will produce
a .spp file for each .gexf file which contains the shortest path patterns of a graph
given the vocabulary of patterns across the dataset.

The algorithms herein are heavily inspired by deep_graph_kernel.py by Yanardag et al 2015

Author: Paul Scherer 2019
"""

# Standard libraries
import sys
import os
import glob
import pickle
import itertools
import random
from collections import defaultdict
from time import time

# 3rd party
import numpy as np
import networkx as nx
import pynauty

# Internal
from decomposition.utils import get_files

# Random seeds from Yanardag et al.
random.seed(314124)
np.random.seed(2312312)

def load_graph(file_handle):
    """
    Loads a numpy adjacency matrix of the GEXF file graph.
    """
    graph = nx.read_gexf(file_handle)
    adj_matrix = nx.to_numpy_matrix(graph)
    return graph, adj_matrix

corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG/"

def save_sp_doc(gexf_fh, gidx, sp_corpus):
	"""
	Saves a shortest path graph doc with the extension .spp
	"""
	open_fname = gexf_fh + ".spp"
	if os.path.isfile(open_fname):
	    return
	with open(open_fname,'w') as fh:
		for pattern in sp_corpus[gidx]:
			print(pattern, file=fh)


def sp_corpus(corpus_dir):
	graph_files = get_files(corpus_dir, extension=".gexf")
	vocabulary = set()
	count_map = {}
	sp_corpus = {}
	corpus = []

	for gexf_fh in graph_files:
		gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
		graph, am = load_graph(gexf_fh)

		count_map[gidx] = {}
		sp_corpus[gidx] = []
		label_map = [graph.nodes[nidx]["Label"] for nidx in sorted(list(graph.nodes()))]

		G = graph
		all_shortest_paths = nx.all_pairs_shortest_path(G) # nx.floyd_warshall(G)
		tmp_corpus = []
		for source, sink_map in list(all_shortest_paths):
			for sink, path in sink_map.items():
				sp_length=len(path)-1
				label = "_".join(map(str, sorted([label_map[int(source)-1], label_map[int(sink)-1]]))) + "_" + str(sp_length)
				tmp_corpus.append(label)
				count_map[gidx][label] = count_map[gidx].get(label,0) + 1
				sp_corpus[gidx].append(label)
				vocabulary.add(label)
		corpus.append(tmp_corpus)

		# Save the document
		save_sp_doc(gexf_fh, gidx, sp_corpus)

	num_graphs = len(count_map)
	return sp_corpus, vocabulary, count_map, num_graphs

if __name__ == '__main__':
	corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG/"
	sp_corpus, vocabulary, count_map, num_graphs = sp_corpus(corpus_dir)