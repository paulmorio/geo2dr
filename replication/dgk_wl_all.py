"""
An example reimplementation of DGK-WL (Yanardag and Vishwanathan 2015) 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import os
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_rbf_bag_of_words

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Output Embeddings folder
output_folder = "DGK_WL_Performance_"+ dataset
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# WL Hyperparameters as in Yanardag and Vishwanathan
wl_depth = 2
min_count = 0

# Skipgram hyperparameters
emb_dimensions = [2,5,10,25,50]
batch_sizes = [10000, 1000] # Same as gensim default in their paper
num_epochs = [5, 100]
initial_lrs = [0.1, 0.01]
cvs = 3

# Quickcheck list
means_accs = []

for emb_dimension in emb_dimensions:
	for batch_size in batch_sizes:
		for epochs in num_epochs:
			for initial_lr in initial_lrs:
				temp_accs = []
				for run in range(cvs):
					# Create embedding signature and check
					output_fh_signature = "_".join([dataset, str(wl_depth), str(emb_dimension), str(batch_size), str(epochs), str(initial_lr), str(run)])
					output_perf_fh = output_folder + "/" + output_fh_signature
					if os.path.exists(output_perf_fh):
						print("%s exists no need to learn embeddings" % (output_perf_fh))
						continue # no need to learn
					else:
						print("Learning embeddings for %s " % (output_perf_fh))

					graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
					corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
					extension = ".wld" + str(wl_depth) # Extension of the graph document

					trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh="temp_embeddings.json",
									  emb_dimension=emb_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr, min_count=0)
					trainer.train()
					final_subgraph_embeddings = trainer.skipgram.give_target_embeddings()


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
					S = (P.dot(M))

					class_labels_fname = "data/"+ dataset + ".Labels"
					# kernel_row_x_id, kernel_row_y_id = utils.get_kernel_matrix_row_idx_with_class(trainer.corpus, extension, graph_files, class_labels_fname)
					xylabels = utils.get_class_labels_tuples(graph_files,class_labels_fname)
					xylabels.sort(key=lambda tup: tup[0])
					kernel_row_x_id, kernel_row_y_id = zip(*xylabels)

					acc, std = cross_val_accuracy_rbf_bag_of_words(S, kernel_row_y_id)
					print ('#... Accuracy score: %0.4f, Standard deviation: %0.4f' % (acc, std))

					with open(output_perf_fh, "w") as fh:
						print("%s,%s" % (str(acc), str(std)), file=fh)

					temp_accs.append(acc)

				if not np.isnan(np.mean(temp_accs)):
					means_accs.append(np.mean(temp_accs))

avg_max = np.max(means_accs)
print("The best average of average scores was: %s" % (np.max(means_accs)))

with open("best_wl", "w") as fh:
	print("%s" % (avg_max), file=fh)