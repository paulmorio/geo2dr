"""
An example reimplementation of DGK-WL (Yanardag and Vishwanathan 2015) 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import os
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.graphlet_patterns import graphlet_corpus
from geometric2dr.embedding_methods.skipgram_trainer import Trainer, InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_rbf_bag_of_words

from gensim.models import Word2Vec

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Output Embeddings folder
output_folder = "DGK_GK_Performance_"+ dataset
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# WL Hyperparameters as in Yanardag and Vishwanathan
num_graphlets = [7,8]
sample_sizes = [2,5,10,25,50]

# Skipgram hyperparameters
emb_dimensions = [2,5,10,25,50]
batch_sizes = [10000, 1000] # Same as gensim default in their paper
num_epochs = [5, 100]
cvs = 3

def l2_norm(vec):
	return  np.sqrt(np.dot(vec, vec))

# Quickcheck list
means_accs = []

for num_graphlet in num_graphlets:
	for sample_size in sample_sizes:
		for emb_dimension in emb_dimensions:
			for batch_size in batch_sizes:
				for epochs in num_epochs:
					temp_accs = []
					for run in range(cvs):
						# Create embedding signature and check
						output_fh_signature = "_".join([dataset, str(num_graphlet), str(sample_size), str(emb_dimension), str(batch_size), str(epochs), str(run)])
						output_perf_fh = output_folder + "/" + output_fh_signature
						if os.path.exists(output_perf_fh):
							print("%s exists no need to learn embeddings" % (output_perf_fh))
							continue # no need to learn
						else:
							print("Learning embeddings for %s " % (output_perf_fh))

						graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
						corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_data_dir, num_graphlet, sample_size)
						extension = ".graphlet_ng_"+str(num_graphlet)+"_ss_"+str(sample_size)

						corpus = [[str(x) for x in group] for group in corpus]

						model = Word2Vec(corpus, size=emb_dimension, window=10, min_count=0, sg=1, hs=0, iter=epochs, negative=10, batch_words=batch_size)

						# deep w/l2 norm
						P = np.zeros((num_graphs, len(vocabulary)))
						for i in range(num_graphs):
							for jdx, j in enumerate(vocabulary):
								P[i][jdx] = prob_map[i+1].get(j,0)
						M = np.zeros((len(vocabulary), len(vocabulary)))
						for idx,i in enumerate(vocabulary):
							M[idx][idx] = l2_norm(model[str(i)])
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

with open("best_gk", "w") as fh:
	print("%s" % (avg_max), file=fh)