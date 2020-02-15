"""
Experimental script which run, sets up and learns distributed representations
of subgraphs, then uses them to create a kernel matrix as in Deep Graph Kernels
by Yanardag and Vishwanathan

This version uses graphlets as its main pattern that is induced in the graphs.

Hyperparameters are decomposition algorithm and skipgram specific. Graphlets have the
following hyperparameters
- num_graphlets
- samplesize

skipgram has the following hyperparameters
- embedding_dimensions
- batch_sizes
- epochs
- initial_learning rate(fixed for now)
- min_count (on number of subgraph patterns)

Author: Paul Scherer
"""

import os
import numpy as np

import embedding_methods.utils as utils
from decomposition.shortest_path_patterns import sp_corpus
from embedding_methods.skipgram_trainer import Trainer
from embedding_methods.classify import cross_val_accuracy_precomputed_kernel_matrix

# Skipgram hyperparameters 
min_counts = [0]
embedding_dimensions = [25, 50]
batch_size = 256
runs = 2
epochs = 3
initial_lr = 0.1

embedding_folder = "embeddings"
dataset = "MUTAG"
corpus_dir = "data/dortmund_gexf/" + dataset

def l2_norm(vec):
	return  np.sqrt(np.dot(vec, vec))

extension = ".spp"
for embedding_dimension in embedding_dimensions:
	for min_count in min_counts:
		for run in range(runs):
			output_fh = str.join("_", [dataset, "SubEmbeddings_SP", "embd", str(embedding_dimension), "epochs", str(epochs), "bs", str(batch_size), "minCount", str(min_count), "run", str(run)])
			output_fh += ".json"
			output_fh = embedding_folder + "/" + output_fh
			# No need to learn something that exists.
			if os.path.exists(output_fh):
				print("Embedding %s exists continuing..." % (output_fh))
				continue

			# Run the decomposition algorithm
			graph_files = utils.get_files(corpus_dir, ".gexf", max_files=0)
			print("###.... Loaded %s files in total" % (str(len(graph_files))))
			print(output_fh)
			corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_dir)

			trainer = Trainer(corpus_dir=corpus_dir, extension=extension, max_files=0, window_size=10, output_fh=output_fh,
							  emb_dimension=embedding_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
							  min_count=min_count)
			trainer.train()
			final_embeddings = trainer.skipgram.give_target_embeddings()

			# Compute the kernel matrix for the graphs using our substructure pattern embeddings
			kernel_type = 3
			K = np.zeros((num_graphs, num_graphs))
			# P = np.zeros((num_graphs, vocab_size))
			if kernel_type == 1:
				# Using substructure embeddings with l2 norm
				vocabulary = list(sorted(trainer.corpus._subgraph_to_id_map.keys())) # because the vocabulary is changed by the window size
				vocab_size = trainer.vocab_size
				P = np.zeros((num_graphs, vocab_size))
				for i in range(num_graphs):
					for jdx, j in enumerate(vocabulary):
						P[i][jdx] = prob_map[i+1].get(j,0) # prob map already maps proper gidx of gexf file 
				M = np.zeros((len(vocabulary), len(vocabulary)))
				for idx, i in enumerate(vocabulary):
					M[idx][idx] = l2_norm(final_embeddings[trainer.corpus._subgraph_to_id_map[str(i)]])
				K = (P.dot(M)).dot(P.T) # rows of kernel matrices have to swapped according to id_to_graph map or sort the y values
			
			elif kernel_type == 2:
				vocabulary = list(sorted(trainer.corpus._subgraph_to_id_map.keys())) # because the vocabulary is changed by the window size
				vocab_size = trainer.vocab_size
				P = np.zeros((num_graphs, trainer.vocab_size))
				for i in range(num_graphs):
					for jdx, j in enumerate(vocabulary):
						P[i][jdx] = prob_map[i+1].get(j,0) # prob map already maps proper gidx of gexf file 
				M = np.zeros((len(vocabulary), len(vocabulary)))
				for i in range(len(vocabulary)):
					for j in range(len(vocabulary)):
						M[i][j] = np.dot(final_embeddings[trainer.corpus._subgraph_to_id_map[str(vocabulary[i])]], final_embeddings[trainer.corpus._subgraph_to_id_map[str(vocabulary[j])]])
				K = (P.dot(M)).dot(P.T)
			
			elif kernel_type == 3:
				# Simple MLE Kernel which does not use substructure embeddings
				vocab_size = len(vocabulary)
				vocabulary = list(sorted(vocabulary))
				P = np.zeros((num_graphs, vocab_size))
				print(vocab_size)
				for i in range(num_graphs):
					for jdx, j in enumerate(vocabulary):
						P[i][jdx] = prob_map[i+1].get(j,0)
				K = P.dot(P.T)


			class_labels_fname = "data/"+ dataset + ".Labels"
			kernel_row_x_id, kernel_row_y_id = utils.get_kernel_matrix_row_idx_with_class(trainer.corpus, extension, graph_files, class_labels_fname)

			acc, std = cross_val_accuracy_precomputed_kernel_matrix(K, kernel_row_y_id)
			print ('accuracy score: %0.3f, standard deviation: %0.3f' % (acc, std))
