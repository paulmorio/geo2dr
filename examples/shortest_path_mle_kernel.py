"""
An example of Shortest Paths MLE Kernel
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.shortest_path_patterns import sp_corpus
from geometric2dr.embedding_methods.skipgram_trainer import Trainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_precomputed_kernel_matrix

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir) # will produce .spp files

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
		P[i][jdx] = prob_map[i+1].get(j,0)
K = P.dot(P.T)

class_labels_fname = "data/"+ dataset + ".Labels"
# kernel_row_x_id, kernel_row_y_id = utils.get_kernel_matrix_row_idx_with_class(trainer.corpus, extension, graph_files, class_labels_fname)
xylabels = utils.get_class_labels_tuples(graph_files,class_labels_fname)
xylabels.sort(key=lambda tup: tup[0])
kernel_row_x_id, kernel_row_y_id = zip(*xylabels)

acc, std = cross_val_accuracy_precomputed_kernel_matrix(K, kernel_row_y_id)
print ('#... Accuracy score: %0.3f, Standard deviation: %0.3f' % (acc, std))