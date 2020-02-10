"""
Graphlet based graph decomposition algorithm to create graph documents

The main use case is for the user to input a path containing individual
graphs of the dataset in gexf format.
The decomposition algorithm will induce graphlet patterns for graphs
recording the dataset/"global" vocabulary of patterns within a dictionary.
The graph and its associated patterns (by IDs given through our hash function)
are saved into a  <graphid>.wldr<depth> file which contains a line delimited
list of all the substructure pattern ids.

Inspired by deep_graph_kernel.py by Pinar Yanardag 2015
Author: Paul Scherer 2019.
"""

# Sytem libraries
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
import pynauty # make sure to install this

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


def get_maps(num_graphlets):
    with open("decomposition/canonical_maps/canonical_map_n%s.p"%(num_graphlets), 'rb') as handle:
        # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
        canonical_map = pickle.load(handle, encoding="latin1")

    with open("decomposition/graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%(num_graphlets), 'rb') as handle:
        # weight map -> {parent id: {child1: weight1, ...}}
        weight_map = pickle.load(handle, encoding="latin1")

    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.items()} for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k,v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map

def get_graphlet(window, nsize):
    """
    This function takes the upper triangle of a nxn matrix and computes its canonical map
    """
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict = adj_mat)
    cert = pynauty.certificate(g)
    return cert

def graphlet_corpus(corpus_dir, num_graphlets, samplesize):
    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    canonical_map, weight_map = get_maps(num_graphlets)
    stat_arr = []
    vocabulary = set()
    corpus = []

    # randomly sample graphlets
    graph_map = {}
    graph_map2 = {}

    graph_files = get_files(corpus_dir, extension=".gexf")

    for gexf_fh in graph_files:
        gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
        nx_graph, am = load_graph(gexf_fh)
        m = len(am)
        count_map = {} # graphlet countmap
        tmp_corpus = [] # temporary corpus for a single graph

        if m >=num_graphlets:
            for j in range(samplesize):
                r =  np.random.permutation(range(m))
                for n in [num_graphlets]:
                    window = am[np.ix_(r[0:n],r[0:n])]
                    g_type = canonical_map[str(get_graphlet(window, n), 'latin1')] # need str to handle weird encoding from py2
                    graphlet_idx = g_type["idx"]
                    level = g_type["n"]
                    count_map[graphlet_idx] = count_map.get(graphlet_idx, 0) + 1
                    # for each node, pick a node it is connected to, and place a non-overlapping window
                    tmp_corpus.append(graphlet_idx)
                    vocabulary.add(graphlet_idx)
                    for node in r[0:n]:
                        # place a window to for each node in the original window
                        new_n_arr = r[n:][0:n-1] # select n-1 nodes
                        r2 = np.array(list(new_n_arr) + [node])
                        window2 = am[np.ix_(r2,r2)]
                        g_type2 = canonical_map[str(get_graphlet(window2, n), 'latin1')]
                        graphlet_idx2 = g_type2["idx"]
                        count_map[graphlet_idx2] = count_map.get(graphlet_idx2, 0) + 1
                        vocabulary.add(graphlet_idx2)
                        tmp_corpus.append(graphlet_idx2)
                    corpus.append(tmp_corpus)
        else:
            count_map[fallback_map[num_graphlets]] = samplesize # fallback to 0th node at that level
        graph_map[gidx] = count_map
        # print ("Graph: %s #nodes: %s  total samples: %s" % (gidx, len(nx_graph.nodes()), sum(list(graph_map[gidx].values()))))
        save_graphlet_document(gexf_fh, gidx, graph_map, num_graphlets, samplesize)

    print ("Total size of the corpus: %s" % (len(corpus)))
    prob_map = {gidx: {graphlet: count/float(sum(graphlets.values())) \
        for graphlet, count in graphlets.items()} for gidx, graphlets in graph_map.items()}
    num_graphs = len(prob_map)

    return corpus, vocabulary, prob_map, num_graphs, graph_map

def save_graphlet_document(gexf_fh, gidx, graph_map, num_graphlets, samplesize):
    open_fname = gexf_fh + ".graphlet" + "_ng_" + str(num_graphlets) + "_ss_" + str(samplesize)
    if os.path.isfile(open_fname):
        return
    with open(open_fname,'w') as fh:
        for graphlet_pattern in list(graph_map[gidx].keys()):
            # write the pattern the count number of times
            for _ in range(graph_map[gidx][graphlet_pattern]):
                sentence=str(graph_map[gidx][graphlet_pattern])
                print(sentence, file=fh)


if __name__ == '__main__':
    corpus_dir = corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG/"
    corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_dir, 5, 6)