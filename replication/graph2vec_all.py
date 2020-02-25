"""
A script that uses the Geo2DR modules to recreate Graph2Vec's
Narayanan 2017 original graph classificaition setup and run 
against arbitrary datasets.

Author: Paul Scherer 2019
"""

import os

from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
from geometric2dr.embedding_methods.pvdbow_trainer import InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy
import geometric2dr.embedding_methods.utils as utils

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Output Embeddings folder
output_folder = "Graph2Vec_Embeddings"
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# Graph2Vec Hyperparameters
wl_depths = [2,3]
min_count = 0

# PVDBOW Hyperparameters
emb_dimensions = [8, 16, 32, 64, 128]
batch_sizes = [128, 256, 512]
num_epochs = [25, 50, 100]
initial_lrs = [0.1, 0.01, 0.001]
cvs = 10


# # Graph2Vec Hyperparameters
# wl_depths = [2]
# min_count = 0

# # PVDBOW Hyperparameters
# emb_dimensions = [8, 16]
# batch_sizes = [128]
# num_epochs = [5, 10]
# initial_lrs = [0.1]
# cvs = 10

for wl_depth in wl_depths:
	for emb_dimension in emb_dimensions:
		for batch_size in batch_sizes:
			for epochs in num_epochs:
				for initial_lr in initial_lrs:
					for run in range(cvs):
						# Create embedding signature and check
						output_fh_signature = "_".join([dataset, str(wl_depth), str(emb_dimension), str(batch_size), str(epochs), str(initial_lr), str(run)])
						output_embedding_fh = output_folder + "/" + output_fh_signature
						if os.path.exists(output_embedding_fh):
							print("%s exists no need to learn embeddings" % (output_embedding_fh))
							continue # no need to learn
						else:
							print("Learning embeddings for %s " % (output_embedding_fh))

						# Learn embeddings
						graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
						wl_corpus(graph_files, wl_depth)
						extension = ".wld" + str(wl_depth) # Extension of the graph document

						trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=output_embedding_fh,
						                  emb_dimension=emb_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
						                  min_count=min_count)
						trainer.train()