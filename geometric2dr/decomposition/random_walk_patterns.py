"""Functions for inducing random walk patterns within the graphs
as in Gartner et al.

Inspired by the implementation of anonymous walks by Sergey Ivanov
and Evgeny Burnaev [1]_

"""

# Author: Paul Scherer 2020

# Standard libraries
import os
import random
from tqdm import tqdm

# 3rd party
import numpy as np
import networkx as nx

# Internal 
from .utils import get_files

# Random seeds for reproduction
random.seed(1234)
np.random.seed(1234)

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


def random_walks_graph(nx_graph, walk_length, neighborhood_size=0):
	"""Perform a random walk with cooccurring "neighbouring" walks
	starting from the same node for all the nodes in the graph.
	"""
	neigh_size = neighborhood_size + 1 # there is at least one random walk per node
	rw_graph = create_random_walk_graph(nx_graph)
	walks = []
	for i, node in enumerate(rw_graph):
		rw = [str(random_walk_with_label_nodes(nx_graph, rw_graph, node, walk_length)) for _ in range(neighborhood_size)]  # for anonymous_walk
		walks.append(rw)
	return walks

def save_rw_doc(gexf_fh, coocurrence_corpus, walk_length):
	"""Saves a shortest path graph doc with the extension .spp
	
	Parameters
	----------
	gexf_fh : str
		sdf
	cooccurrence_corpus : list
		list of lists containing cooccuring shortest paths 
		(ie those starting from the same starting node)
	walk_length : int
		length of the walk

	Returns
	-------
	None : None
		Saves the induced shortest path patterns into a graph document for the graph specified
		in the `gexf_fh` 

	"""

	open_fname = gexf_fh + ".rw"  + str(walk_length)
	# if os.path.isfile(open_fname):
	# 	return
	with open(open_fname,'w') as fh:
		for spp_neighbourhood in coocurrence_corpus:
			sentence = str.join(" ", map(str, spp_neighbourhood))
			print (sentence, file=fh)

def rw_corpus(corpus_dir, walk_length, neighborhood_size, saving_graph_docs=True):
	"""Induces random walks up to a desired length across the input set of graphs
	
	Parameters
	----------
	corpus_dir : str
		path to directory containing graph files
	walk_length : int
		desired length of the random walk
	neighborhood_size : int
		number of cooccuring walks to find per walk source (node)
	saving_graph_docs : bool
		boolean on whether the graph documents should be generated or not

	Returns
	-------
	None : None
		Induces the random walks of `walk_length` across the list of graphs files
		supplied in `corpus_dir` 
	"""

	graph_files = get_files(corpus_dir, extension=".gexf")
	vocabulary = set()
	count_map = {}
	graph_map = {}
	corpus = []
	pattern_to_short_id = {}
	short_id_to_patern = {}
	pattern_id = 0

	for gexf_fh in tqdm(graph_files):
		gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
		graph, am = load_graph(gexf_fh)

		tmp_corpus = []
		count_map[gidx] = {}
		cooccurence_corpus = []
		
		# Induce random walks across the single graph
		rws_graph = random_walks_graph(graph, walk_length, neighborhood_size)
		for node_walks in rws_graph:
			paths_cooccurence = []
			for source_wlk in node_walks:
				# make a shorter label for each pattern
				if source_wlk in pattern_to_short_id:
					label = pattern_to_short_id[source_wlk]
				else:
					pattern_to_short_id[source_wlk] = str(pattern_id)
					short_id_to_patern[str(pattern_id)] = source_wlk
					label = pattern_to_short_id[source_wlk]
					pattern_id += 1

				tmp_corpus.append(label)
				paths_cooccurence.append(label)
				count_map[gidx][label] = count_map[gidx].get(label,0)+1
				vocabulary.add(label)
			cooccurence_corpus.append(paths_cooccurence)

		corpus.append(tmp_corpus)
		if saving_graph_docs:
			save_rw_doc(gexf_fh, cooccurence_corpus, walk_length)

	graph_map = count_map
	prob_map = {gidx: {path: count/float(sum(paths.values())) \
		for path, count in paths.items()} for gidx, paths in count_map.items()}

	num_graphs = len(count_map)
	return corpus, vocabulary, prob_map, num_graphs, graph_map

if __name__ == "__main__":
	corpus_dir = "../data/dortmund_gexf/MUTAG/"
	corpus, vocabulary, prob_map, num_graphs, graph_map = rw_corpus(corpus_dir, 4, 5)