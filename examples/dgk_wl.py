"""
An example reimplementation of DGK-WL (Yanardag and Vishwanathan 2015) 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.weisfeiler_lehman_patterns import wlk_relabeled_corpus
from geometric2dr.embedding_methods.skipgram_trainer import Trainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_precomputed_kernel_matrix

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths for subgraph embeddings
output_embedding_fh = "WL_Subgraph_Embeddings.json"

# WL decomposition hyperparameters
wl_depth = 2

############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
wlk_relabeled_corpus(graph_files, wl_depth)
extension = ".wld" + str(wl_depth) # Extension of the graph document

############
# Step 2
# Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
############
trainer = Trainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh=output_embedding_fh,
				  emb_dimension=32, batch_size=128, epochs=50, initial_lr=0.1, min_count=1)
trainer.train()
final_subgraph_embeddings = trainer.skipgram.give_target_embeddings()

############
# Step 3
# Create a kernel matrix of the graphs using the embeddings of the substructures
############
num_graphs = trainer.corpus.num_graphs
K = np.zeros((num_graphs, num_graphs))
vocabulary = list(sorted(trainer.corpus._subgraph_to_id_map.keys())) # Use the vocabulary used in training embeddings
vocab_size = trainer.vocab_size
P = np.zeros((num_graphs, trainer.vocab_size))
for i in range(num_graphs):
	for jdx, j in enumerate(vocabulary):
		P[i][jdx] = prob_map[i+1].get(j,0)
M = np.zeros((len(vocabulary), len(vocabulary)))
for i in range(len(vocabulary)):
	for j in range(len(vocabulary)):
		M[i][j] = np.dot(final_subgraph_embeddings[trainer.corpus._subgraph_to_id_map[str(vocabulary[i])]], final_subgraph_embeddings[trainer.corpus._subgraph_to_id_map[str(vocabulary[j])]])
K = (P.dot(M)).dot(P.T)

###########
# Step 4
# Use some kernel method, such as an SVM to compute classifications on the graph kernel matrix
###########
class_labels_fname = "data/"+ dataset + ".Labels"
kernel_row_x_id, kernel_row_y_id = utils.get_kernel_matrix_row_idx_with_class(trainer.corpus, extension, graph_files, class_labels_fname)
acc, std = cross_val_accuracy_precomputed_kernel_matrix(K, kernel_row_y_id)
print ('#... Accuracy score: %0.3f, Standard deviation: %0.3f' % (acc, std))