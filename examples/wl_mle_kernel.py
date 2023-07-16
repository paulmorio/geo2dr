"""
An example reimplementation of DGK-WL (Yanardag and Vishwanathan 2015) 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_rbf_bag_of_words

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# WL decomposition hyperparameters
wl_depth = 2

############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)

############
# Step 2
# Compute the kernel and use a kernel method to perform classification
############
# Simple MLE Kernel which does not use substructure embeddings
vocab_size = len(vocabulary)
vocabulary = list(sorted(vocabulary))
P = np.zeros((num_graphs, vocab_size))
for i in range(num_graphs):
	for jdx, j in enumerate(vocabulary):
		P[i][jdx] = prob_map[str(i+1)].get(j,0)
K = P.dot(P.T)

class_labels_fname = "data/"+ dataset + ".Labels"
xylabels = utils.get_class_labels_tuples(graph_files,class_labels_fname)
xylabels.sort(key=lambda tup: tup[0])
kernel_row_x_id, kernel_row_y_id = zip(*xylabels)

acc, std = cross_val_accuracy_rbf_bag_of_words(P, kernel_row_y_id)
print ('#... Accuracy score: %0.3f, Standard deviation: %0.3f' % (acc, std))