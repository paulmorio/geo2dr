"""
An example reimplementation of Graph2Vec (Narayanan et al) 
using Geo2DR (geometric2dr)

Author: Paul Scherer 2020
"""

import os

from geometric2dr.decomposition.weisfeiler_lehman_patterns import wlk_relabeled_corpus
from geometric2dr.embedding_methods.pvdbow_trainer import InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy
import geometric2dr.embedding_methods.utils as utils

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths
output_embedding_fh = "Graph2Vec_Embeddings.json"

# Hyper parameters
wl_depth = 2
min_count_patterns = 0 # min number of occurrences to be considered in vocabulary of subgraph patterns


#######
# Step 1 Create corpus data for neural language model
# We keep permanent files for sake of deeper post studies and testing
#######
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
wlk_relabeled_corpus(graph_files, wl_depth)
extension = ".wld" + str(wl_depth) # Extension of the graph document

######
# Step 2 Train a neural language model to learn distributed representations
# 		 of the graphs directly or of its substructures. Here we learn it directly
#		 for an example of the latter check out the DGK models.
######
# Instantiate a PV-DBOW trainer to learn distributed reps directly.
trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=output_embedding_fh,
                  emb_dimension=32, batch_size=128, epochs=250, initial_lr=0.1,
                  min_count=min_count_patterns)
trainer.train()

#######
# Step 3?: Downstream processing. In this case we will just do graph classification using an SVM to perform 10 fold CV
#######
final_embeddings = trainer.skipgram.give_target_embeddings()
graph_files = trainer.corpus.graph_fname_list
class_labels_fname = "data/"+ dataset + ".Labels"
classify_scores = cross_val_accuracy(corpus_dir=corpus_data_dir, extension=trainer.corpus.extension, embedding_fname=trainer.output_fh, class_labels_fname=class_labels_fname)
mean_acc, std_dev = classify_scores
print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (mean_acc, std_dev))