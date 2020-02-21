"""
An example implementation of AWE-DD  (Burnaev and Ivanov) 
using shortest paths instead of anonymous walks 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""
import os

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.shortest_path_patterns import sp_corpus
from geometric2dr.embedding_methods.classify import cross_val_accuracy
from geometric2dr.embedding_methods.pvdm_trainer import PVDM_Trainer # Note use of PVDM


# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths
output_embedding_fh = "WL_PVDM_Embeddings.json"

#######
# Step 1 Create corpus data for neural language model
# We keep permanent files for sake of deeper post studies and testing
#######
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir) # will produce .spp files
extension = ".spp"

cumacc = []
for _ in range(10):
	######
	# Step 2 Train a neural language model to learn distributed representations
	# 		 of the graphs directly or of its substructures. Here we learn it directly
	#		 for an example of the latter check out the DGK models.
	######
	trainer = PVDM_Trainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh=output_embedding_fh,
	                  emb_dimension=32, batch_size=100, epochs=15, initial_lr=0.001, min_count=1)
	trainer.train()

	#######
	# Step 3?: Downstream processing. In this case we will just do graph classification using an SVM to perform 10 fold CV
	#######
	final_embeddings = trainer.pvdm.give_target_embeddings()
	graph_files = trainer.corpus.graph_fname_list
	class_labels_fname = "data/"+ dataset + ".Labels"
	classify_scores = cross_val_accuracy(corpus_dir=corpus_data_dir, extension=trainer.corpus.extension, embedding_fname=trainer.output_fh, class_labels_fname=class_labels_fname)
	mean_acc, std_dev = classify_scores
	print("SVM Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))

	cumacc.append(mean_acc)

import numpy as np
print("Overall SP-PVDM + SVM Mean accuracy using 10 cross fold accuracy: %s with std %s" % (np.mean(cumacc), np.std(cumacc)))