"""Shortest path based decomposition algorithm to create graph documents. 
Inspired and adapted from Yanardag and Vishwanathan "Deep Graph Kernels" [2]_.

.. [2]  P. Yanardag and S. Vishwanathan, "Deep Graph Kernels", KDD '15: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015

"""

# Copyright (c) 2016 Pinar Yanardag
#               2019 Paul Scherer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

# Internal
from .utils import get_files

# Random seeds from Yanardag et al.
random.seed(314124)
np.random.seed(2312312)

def load_graph(file_handle):
	""" Load the gexf format file as a nx_graph and an adjacency matrix.

	Parameters
	----------
	file_handle : str
	    path to gexf file

	Returns
	-------
	graph : networkx graph
	    Networkx graph instance of input gexf file
	adj_matrix : numpy ndarray
	    The adjacency matrix of the graph

	"""
	graph = nx.read_gexf(file_handle)
	adj_matrix = nx.to_numpy_matrix(graph)
	return graph, adj_matrix

def save_sp_doc(gexf_fh, gidx, coocurrence_corpus):
	"""Saves a shortest path graph doc with the extension .spp
	
	Parameters
	----------
	gexf_fh : str
		sdf
	gidx : int
		int id of the gexf (note: deprecated and will 
		be removed in next version)
	cooccurrence_corpus : list
		list of lists containing cooccuring shortest paths 
		(ie those starting from the same starting node)

	Returns
	-------
	None : None
		Saves the induced shortest path patterns into a graph document for the graph specified
		in the `gexf_fh` 

	"""
	
	open_fname = gexf_fh + ".spp"
	# if os.path.isfile(open_fname):
	# 	return
	with open(open_fname,'w') as fh:
		for spp_neighbourhood in coocurrence_corpus:
			sentence = str.join(" ", map(str, spp_neighbourhood))
			print (sentence, file=fh)

def sp_corpus(corpus_dir):
	"""
	The main use case for this script is for the user to supply a path to the directory
	containing the .gexf graph files of a dataset. The decomposition function will produce
	a .spp file for each .gexf file which contains the shortest path patterns of a graph
	given the vocabulary of patterns across the dataset.
	"""
	graph_files = get_files(corpus_dir, extension=".gexf")
	vocabulary = set() 
	count_map = {}
	graph_map = {}
	corpus = []

	for gexf_fh in graph_files:
		open_fname = gexf_fh + ".spp"
		if os.path.exists(open_fname):
			continue
		
		gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
		graph, am = load_graph(gexf_fh)

		count_map[gidx] = {}
		label_map = [graph.nodes[nidx]["Label"] for nidx in sorted(list(graph.nodes()))]
		coocurrence_corpus = []

		G = graph
		all_shortest_paths = nx.all_pairs_shortest_path(G)
		tmp_corpus = []
		
		# For each source and all the path endpoints add the label
		# We (Yanardag and subsequently Burnaev) consider paths to 
		# "cooccur" if they share the same source node
		for source, sink_map in list(all_shortest_paths):
			paths_cooccurrence = []
			for sink, path in sink_map.items():
				sp_length=len(path)-1
				label = "_".join(map(str, sorted([label_map[int(source)-1], label_map[int(sink)-1]]))) + "_" + str(sp_length)
				tmp_corpus.append(label)
				paths_cooccurrence.append(label)
				count_map[gidx][label] = count_map[gidx].get(label,0) + 1
				vocabulary.add(label)
			coocurrence_corpus.append(paths_cooccurrence)
		
		# Record and save the information
		save_sp_doc(gexf_fh, gidx, coocurrence_corpus)
		corpus.append(tmp_corpus)

	graph_map = count_map
	# When we are using a straight up MLE kernel we use the graph_map with the
	# counts instead
	prob_map = {gidx: {path: count/float(sum(paths.values())) \
		for path, count in paths.items()} for gidx, paths in count_map.items()}

	num_graphs = len(count_map)
	return corpus, vocabulary, prob_map, num_graphs, graph_map

if __name__ == '__main__':
	corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/data/dortmund_gexf/MUTAG/"
	corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_dir)