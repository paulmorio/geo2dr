"""Graphlet based graph decomposition algorithm to create graph documents. 
Inspired and adapted from Yanardag and Vishwanathan "Deep Graph Kernels" [2]_.

.. [2]  P. Yanardag and S. Vishwanathan, "Deep Graph Kernels", KDD '15: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015

"""

# We adopt the original license and extend it here.
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
from .utils import get_files

# Random seeds from Yanardag et al.
random.seed(314124)
np.random.seed(2312312)


def load_graph(file_handle):

    graph = nx.read_gexf(file_handle)
    adj_matrix = nx.to_numpy_matrix(graph)
    return graph, adj_matrix

def get_maps(num_graphlets):


    data_path = os.path.join(os.path.dirname(__file__), 'canonical_maps')
    # data_path = "canonical_maps"
    with open(data_path + "/canonical_map_n%s.p"%(num_graphlets), 'rb') as handle:
        # canonical_map : {canonical string id: {"graph", "idx", "n"}}
        canonical_map = pickle.load(handle, encoding="latin1")
    return canonical_map

def get_graphlet(window, num_graphlets):

    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=num_graphlets, directed=False, adjacency_dict = adj_mat)
    cert = pynauty.certificate(g)
    return cert

def graphlet_corpus(corpus_dir, num_graphlets, samplesize):


    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    canonical_map = get_maps(num_graphlets)
    vocabulary = set()
    corpus = []

    # randomly sample graphlets
    graph_map = {}
    graph_map2 = {}

    graph_files = get_files(corpus_dir, extension=".gexf")

    for gexf_fh in graph_files:
        gidx = int((os.path.basename(gexf_fh)).replace(".gexf", ""))
        nx_graph, adj_mat = load_graph(gexf_fh)
        num_nodes = len(adj_mat)

        count_map = {} # graphlet countmap
        tmp_corpus = [] # temporary corpus of graphlet patterns for a single graph
        cooccurence_corpus = [] # corpus which preserves cooccurence as in Yanardag et al.

        # Only sample graphlets if the number of nodes in the graph is larger than the
        # maximum graphlet size
        if num_nodes >= num_graphlets:
            for _ in range(samplesize):
                r =  np.random.permutation(range(num_nodes))
                for n in [num_graphlets]:
                    # "Main Graphlet"
                    # Choose a random set of num_graphlet nodes, find the graphlets of 
                    # desired size and add it to the count_map
                    window = adj_mat[np.ix_(r[0:n],r[0:n])]
                    g_type = canonical_map[str(get_graphlet(window, n), 'latin1')]
                    graphlet_idx = g_type["idx"]
                    count_map[graphlet_idx] = count_map.get(graphlet_idx, 0) + 1
                    vocabulary.add(graphlet_idx)
                    tmp_corpus.append(graphlet_idx)
                    cooccurence_corpus.append([graphlet_idx])

                    # "Co-occuring Graphlets"
                    for node in r[0:n]:
                        # select a non-overlapping window n-1 in size and one of 
                        # the nodes in the "main graphlet", to find a neighbouring graphlet
                        new_n_arr = r[n:][0:n-1]
                        r2 = np.array(list(new_n_arr) + [node])
                        window2 = adj_mat[np.ix_(r2,r2)]
                        g_type2 = canonical_map[str(get_graphlet(window2, n), 'latin1')]
                        graphlet_idx2 = g_type2["idx"]
                        count_map[graphlet_idx2] = count_map.get(graphlet_idx2, 0) + 1
                        vocabulary.add(graphlet_idx2)
                        tmp_corpus.append(graphlet_idx2)
                        cooccurence_corpus[-1].append(graphlet_idx2)

                    corpus.append(tmp_corpus)
        else:
            count_map[fallback_map[num_graphlets]] = samplesize

        # Record and save graphlet information for the one graph
        graph_map[gidx] = count_map
        save_graphlet_document(gexf_fh, gidx, num_graphlets, samplesize, cooccurence_corpus)

    # Normalise the probabilities of a graphlet in a graph.
    prob_map = {gidx: {graphlet: count/float(sum(graphlets.values())) \
        for graphlet, count in graphlets.items()} for gidx, graphlets in graph_map.items()}
    num_graphs = len(prob_map)

    return corpus, vocabulary, prob_map, num_graphs, graph_map

def save_graphlet_document(gexf_fh, gidx, num_graphlets, samplesize, cooccurence_corpus):
    """Saves the induced graphlet patterns into dataset folder

    Parameters
    ----------
    gexf_fh : str
        asdf
    gidx : int
        asdf
    num_graphlets : int
        asdf

    Returns
    -------
    

    """
    open_fname = gexf_fh + ".graphlet" + "_ng_" + str(num_graphlets) + "_ss_" + str(samplesize)
    with open(open_fname,'w') as fh:
        for graphlet_neighbourhood in cooccurence_corpus:
            sentence = str.join(" ", map(str, graphlet_neighbourhood))
            print (sentence, file=fh)


if __name__ == '__main__':
    corpus_dir = corpus_dir = "../data/dortmund_gexf/MUTAG/"
    corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_dir, num_graphlets=3, samplesize=6)
    # should result in files with 6 lines each with num_graphlets+1 items
