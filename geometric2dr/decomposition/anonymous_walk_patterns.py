"""This module contains algorithms useful for inducing anonymous walks in graphs.

The algorithms are adapted from the original source code by 
Ivanov and Burnaev 2018 "Anonymous Walk Embeddings" [1]_. Original reference 
implementation can be found in: https://github.com/nd7141/AWE

.. [1]  Sergey Ivanov, Evgeny Burnaev. "Anonymous Walk Embeddings". Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2186-2195, 2018. 

"""

# Author: Paul Scherer 2019
# Carry down MIT License

# Standard libraries
import sys
import os
import glob
import pickle
import itertools
import random
from collections import defaultdict
from time import time
from tqdm import tqdm

# 3rd party
import numpy as np
import networkx as nx

# Internal
from .utils import get_files

# Random seeds from reproduction
random.seed(2018)
np.random.seed(2018)


def load_graph(file_handle):
	"""Loads a nx graph object and a corresponding numpy adjacency 
	matrix given a GEXF file graph.

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

def all_paths(paths, steps, keep_last = False):
	'''Get all possible anonymous walks of length up to steps.
	
	Parameters
	----------

	Returns
	-------
	'''
	new_paths = []
	last_step_paths = [[0, 1]]
	for i in range(2, steps+1):
		current_step_paths = []
		for j in range(i + 1):
			for walks in last_step_paths:
				if walks[-1] != j and j <= max(walks) + 1:
					new_paths.append(walks + [j])
					current_step_paths.append(walks + [j])
		last_step_paths = current_step_paths
	# filter only on n-steps walks
	if keep_last:
		new_paths = list(filter(lambda path: len(path) ==  steps + 1, new_paths))
	paths[steps] = new_paths
	return new_paths

def all_paths_edges(paths, steps, keep_last = True):
	'''Get all possible anonymous walks of length up to steps, 
	using edge labels

	'''
	new_paths = []
	last_step_paths = [[]]
	for i in range(0, steps):
		current_step_paths = []
		for j in range(i + 1):
			for walks in last_step_paths:
				if j <= max(walks + [0]) + 1:
					new_paths.append(walks + [j])
					current_step_paths.append(walks + [j])
		last_step_paths = current_step_paths
	if keep_last:
		new_paths = last_step_paths
	paths[steps] = new_paths
	return new_paths

def all_paths_nodes(paths, steps, keep_last = True):
	'''Get all possible anonymous walks of length up to steps, 
	using node labels
	
	'''
	new_paths = []
	last_step_paths = [[0]]
	for i in range(1, steps+1):
		current_step_paths = []
		for j in range(i + 1):
			for walks in last_step_paths:
				if j <= max(walks) + 1:
					new_paths.append(walks + [j])
					current_step_paths.append(walks + [j])
		last_step_paths = current_step_paths
	if keep_last:
		new_paths = last_step_paths
	paths[steps] = new_paths
	return new_paths

def all_paths_edges_nodes(paths, steps, keep_last = True):
	'''Get all possible anonymous walks of length up to steps, 
	using edge-node labels

	'''
	edge_paths = all_paths_edges(paths, steps, keep_last=keep_last)
	node_paths = all_paths_nodes(paths, steps, keep_last=keep_last)
	new_paths = []
	for p1 in edge_paths:
		for p2 in node_paths:
			if len(p2) == len(p1) + 1:
				current_path = [p2[0]]
				for ix in range(len(p1)):
					current_path.append(p1[ix])
					current_path.append(p2[ix+1])
				new_paths.append(current_path)
	paths[steps] = new_paths
	return new_paths


def create_random_walk_graph(nx_graph):
	"""Generate a random walk graph equivalent of a graph as 
	described in anonymous walk embeddings

	"""

	# get name of the label on graph edges (assume all label names are the same)
	label_name = 'weight'

	RW = nx.DiGraph()
	for node in nx_graph:
		edges = nx_graph[node]
		total = float(sum([edges[v].get(label_name, 1)
						   for v in edges if v != node]))
		for v in edges:
			if v != node:
				RW.add_edge(node, v, weight=edges[v].get(label_name, 1) / total)
	rw_graph = RW
	return rw_graph


def random_step_node(rw_graph, node):
	'''Moves one step from the current according to probabilities of 
	outgoing edges. Return next node.

	'''

	r = random.uniform(0, 1)
	low = 0
	for v in rw_graph[node]:
		p = rw_graph[node][v]['weight']
		if r <= low + p:
			return v
		low += p


def random_walk_node(rw_graph, node, steps):
	'''Creates anonymous walk from a node for arbitrary steps.
	Returns a tuple with consequent nodes.

	'''
	
	d = dict()
	d[node] = 0
	count = 1
	walk = [d[node]]
	for i in range(steps):
		v = random_step_node(rw_graph, node)
		if v not in d:
			d[v] = count
			count += 1
		walk.append(d[v])
		node = v
	return tuple(walk)


def random_walk_with_label_nodes(graph, rw_graph, node, steps):
	'''Creates anonymous walk from a node for arbitrary steps with usage of node labels.
	Returns a tuple with consequent nodes.

	'''
	
	d = dict()
	count = 0
	pattern = []
	for i in range(steps + 1):
		label = graph.nodes[node]['Label']
		if label not in d:
			d[label] = count
			count += 1
		pattern.append(d[label])
		node = random_step_node(rw_graph, node)
	return tuple(pattern)


def random_walk_with_label_edges(graph, rw_graph, node, steps):
	'''Creates anonymous walk from a node for arbitrary steps with usage of edge labels.
	Returns a tuple with consequent nodes.

	'''
	
	idx = 0
	pattern = []
	d = dict()
	for i in range(steps):
		v = random_step_node(rw_graph, node)
		label = int(graph[node][v]['Label'])
		if label not in d:
			d[label] = idx
			idx += 1
		pattern.append(d[label])
	return tuple(pattern)


def random_walk_with_label_edges_nodes(graph, rw_graph, node, steps):
	'''Creates anonymous walk from a node for arbitrary steps with usage of edge and node labels.
	Returns a tuple with consequent nodes.

	'''
	
	node_idx = 0
	edge_idx = 0
	pattern = [0]
	node_labels = dict()
	edge_labels = dict()
	for i in range(steps):
		v = random_step_node(rw_graph, node)
		node_label = graph.nodes[node]['Label']
		edge_label = int(graph[node][v]['Label'])
		if node_label not in node_labels:
			node_labels[node_label] = node_idx
			node_idx += 1
		if edge_label not in edge_labels:
			edge_labels[edge_label] = edge_idx
			edge_idx += 1
		pattern.append(node_labels[node_label])
		pattern.append(edge_labels[edge_label])
		node = v
	return tuple(pattern)


def anonymous_walk_node(graph, rw_graph, node, steps, label_setting=None):
	'''Creates anonymous walk from a node.

	'''
	
	if label_setting is None:
		return random_walk_node(rw_graph, node, steps)
	elif label_setting == 'nodes':
		return random_walk_with_label_nodes(graph, rw_graph, node, steps)
	elif label_setting == 'edges':
		return random_walk_with_label_edges(graph, rw_graph, node, steps)
	elif label_setting == 'edges_nodes':
		return random_walk_with_label_edges_nodes(graph, rw_graph, node, steps)


def anonymous_walks(graph, neighborhood_size, walk_ids, awe_length, label_setting):
	"""Generates anonymous walks for a given graph and input hyperparameters
	
	"""
	rw_graph = create_random_walk_graph(graph)
	aws = []
	for i, node in enumerate(rw_graph):
		# aw = [str(walk_ids[random_walk_node(node, steps)]) for _ in range(neighborhood_size)] # for random_walk
		aw = [str(walk_ids[anonymous_walk_node(graph, rw_graph, node, awe_length, label_setting)]) for _ in range(neighborhood_size)]  # for anonymous_walk
		aws.append(aw)
	return aws

def awe_corpus(corpus_dir, awe_length, label_setting, neighborhood_size=10, saving_graph_docs=True):
	"""Induces anonymous walks up to a supplied length across a set of graphs.

	This function extracts anonymous walks of `awe_length` across the graphs
	in a dataset into a vocabulary. The function saves a graphdoc for each graph
	which records string identifiers of the anonymous walks present in each graph

	The main use case for this function is for the user to supply a path to 
	the directory containing the .gexf file which contains .gexf files of
	each graph in a dataset. The decomposition function will produce a file 
	with the extension *.awe_<`awe_length`>_<`label_setting`>*  for each .gexf 
	file which contains the anonymous walks patterns specified by a string hash
	controlled by the vocabulary of patterns across the list of graphs being studied. 

	Parameters
	----------
	corpus_dir : str
		path to folder containing gexf graph files on which the substructure 
		patterns should be induced
	awe_length : int
		desired length of anonymous walk patterns
	label_setting : str
		information of node/edge labels that should be used; this can be None, 'nodes', 'edges', 'edges_nodes'
	neighborhood_size : int (default=10)
		the number of anonymous walks to take from a source node
	saving_graph_docs : bool (default=True)
		boolean value which dictates whether graph documents will be generated and saved. 

	Returns
	-------
	None : None
		Induces the anonymous walks of `awe_length` across the list of gexf file paths
		supplied in `corpus_dir`. Graph "documents" with the extension 
		*.awe_<`awe_length`>_<`label_setting`>* are created for each of the gexf files in the
		same directory containing string identifiers of the patterns induced in each graph.

	"""
	graph_files = get_files(corpus_dir, extension=".gexf")
	awe_corpus = {}

	# Map of possible anonymous walks
	paths = dict()
	if label_setting is None:
		all_paths(paths, awe_length, keep_last=True)
	elif label_setting == 'nodes':
		all_paths_nodes(paths, awe_length, keep_last=True)
	elif label_setting == 'edges':
		all_paths_edges(paths, awe_length, keep_last=True)
	elif label_setting == 'edges_nodes':
		all_paths_edges_nodes(paths, awe_length, keep_last=True)

	walk_ids = dict()
	for i, path in enumerate(paths[awe_length]):
		walk_ids[tuple(path)] = i

	label_suffix = ""
	if label_setting is not None:
		label_suffix = label_setting

	for gexf_fh in tqdm(graph_files):
		open_fname = gexf_fh + ".awe_" + str(awe_length) + "_" + label_suffix
		if os.path.exists(open_fname):
			continue


		gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
		graph, am = load_graph(gexf_fh)

		# the old write corpus rewritten as anonymous walks of a single graph
		aws_graph = anonymous_walks(graph, neighborhood_size, walk_ids, awe_length, label_setting)
		awe_corpus[gidx] = aws_graph

		if saving_graph_docs:
			# Write to the file.
			if label_suffix:
				open_fname = gexf_fh + ".awe_" + str(awe_length) + "_" + label_suffix
			else:
				open_fname = gexf_fh + ".awe_" + str(awe_length) + "_allpaths"
			with open(open_fname, 'w+') as fh:
				for aw in aws_graph:
					fh.write(' '.join(aw) + '\n')

	corpus = []
	vocabulary = set()
	graph_map = {}
	
	for gidx in sorted(awe_corpus.keys()):
		count_map = {}
		tmp_corpus = []
		for node_neighborhood_awalks in awe_corpus[gidx]:
			for pattern_str in node_neighborhood_awalks:
				vocabulary.add(pattern_str)
				count_map[pattern_str] = count_map.get(pattern_str, 0) + 1
				tmp_corpus.append(pattern_str)
		corpus.append(tmp_corpus)
		graph_map[gidx] = count_map

	# Normalise the probabilities of a graphlet in a graph.
	prob_map = {gidx: {graphlet: count/float(sum(anon_walk_patterns.values())) \
		for graphlet, count in anon_walk_patterns.items()} for gidx, anon_walk_patterns in graph_map.items()}
	num_graphs = len(graph_map)
	return corpus, vocabulary, prob_map, num_graphs, graph_map

if __name__ == '__main__':
	corpus_dir = "../data/dortmund_gexf/MUTAG/"
	awe_length = 4
	label_setting = 'nodes'
	corpus, vocabulary, prob_map, num_graphs, graph_map = awe_corpus(corpus_dir, awe_length, label_setting, saving_graph_docs=True)
