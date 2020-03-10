"""
AWE-DD Experiment Replication
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import os

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.anonymous_walk_patterns import awe_corpus
from geometric2dr.embedding_methods.classify import cross_val_accuracy
from geometric2dr.embedding_methods.pvdm_trainer import PVDM_Trainer # Note use of PVDM

# Input data paths
dataset = "REDDIT-BINARY"
corpus_data_dir = "data/" + dataset

# Output Embeddings folder
output_folder = "AWE_Embeddings_" + dataset
if not os.path.exists(output_folder):
	os.mkdir(output_folder)


# AWE Hyperparameters
aw_length = 10
label_setting = "nodes"

# PVDM Hyperparameters as in paper
emb_dimensions = [128]
batch_sizes = [100, 500, 1000, 5000, 10000]
batch_sizes = [1000, 5000, 10000]
batch_sizes = [10000]
num_epochs = [100]
initial_lrs = [0.1]
window_sizes = [8, 16]
cvs = 2

for emb_dimension in emb_dimensions:
	for batch_size in batch_sizes:
		for epochs in num_epochs:
			for initial_lr in initial_lrs:
				for window_size in window_sizes:
					for run in range(cvs):

						# Create embedding signature and check
						output_fh_signature = "_".join([dataset, str(aw_length), str(emb_dimension), str(batch_size), str(epochs), str(initial_lr), str(window_size), str(run)])
						output_embedding_fh = output_folder + "/" + output_fh_signature
						if os.path.exists(output_embedding_fh):
							print("%s exists no need to learn embeddings" % (output_embedding_fh))
							continue # no need to learn
						else:
							print("Learning embeddings for %s " % (output_embedding_fh))

						awe_corpus(corpus_data_dir, aw_length, label_setting, saving_graph_docs=True)
						extension = ".awe_" + str(aw_length) + "_" + label_setting

						######
						# Step 2 Train a neural language model to learn distributed representations
						# 		 of the graphs directly or of its substructures. Here we learn it directly
						#		 for an example of the latter check out the DGK models.
						######
						trainer = PVDM_Trainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=window_size, output_fh=output_embedding_fh,
						                  emb_dimension=emb_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr, min_count=1)
						trainer.train()