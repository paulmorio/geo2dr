"""
Experimental script which runs, sets up and builds distributed representations of graphs using anonymous walks as the main form of graph object pattern to create datasets for distributive modelling

Hyperparameters are decomposition algorithm and skipgram specific
Anonymous walk embeddings are mainly parameterised by the length of their walk, but also whether we want to capture node/edge labels
- awe_length
- label_setting (in one of four options: None, "nodes", "edges", "edges_nodes")

The skipgram has the following general hyperparameters
- embedding_dimensions
- batch_sizes
- epochs
- initial_learning rate(fixed for now)
- min_count (on number of subgraph patterns)

It will also do this <run> number of times for empirical testing of standard deviation

Author: Paul Scherer
"""
import os

import embedding_methods.utils as utils
from decomposition.anonymous_walk_patterns import awe_corpus
from embedding_methods.classify import cross_val_accuracy
from embedding_methods.pvdm_trainer import PVDM_Trainer

# Decomposition algorithm specfic
awe_lengths = [10] # paper says this is what they used
label_setting = "nodes"

# Paper Replication settings
# Skipgram specific
min_counts = [1] 
embedding_dimensions = [256] # paper spec
batch_size = 100
runs = 5
epochs = 100
initial_lr = 0.01

embedding_folder = "embeddings"
dataset = "MUTAG"
corpus_dir = "data/dortmund_gexf/" + dataset

for embedding_dimension in embedding_dimensions:
	for min_count in min_counts:
		for run in range(runs):
			for awe_length in awe_lengths:
				extension = ".awe_" + str(awe_length) + "_" + label_setting

				# Set up embeddings files
				output_fh = str.join("_", [dataset, "AWE", "aweLength", str(awe_length), "label_setting" ,"embd", str(embedding_dimension), "epochs", str(epochs), "bs", str(batch_size), "minCount", str(min_count), "run", str(run)])
				output_fh += ".json"
				output_fh = embedding_folder + "/" + output_fh

				# No need to learn something that exists.
				if os.path.exists(output_fh):
					print("Embedding %s exists continuing..." % (output_fh))
					continue

				# Run the decomposition algorithm and create graph docs
				graph_files = utils.get_files(corpus_dir, ".gexf", max_files=0)
				print("###.... Loaded %s files in total" % (str(len(graph_files))))
				print(output_fh)
				awe_corpus(corpus_dir, awe_length, label_setting, saving_graph_docs=True)

				# Use the graph documents in the directory to create a trainer which handles creation of datasets/corpus/dataloaders
				# and the skipgram model.
				trainer = PVDM_Trainer(corpus_dir=corpus_dir, extension=extension, max_files=0, window_size=7, output_fh=output_fh,
				                  emb_dimension=embedding_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
				                  min_count=min_count)
				trainer.train()

				# Classification if needed
				final_embeddings = trainer.pvdm.give_target_embeddings()
				graph_files = trainer.corpus.graph_fname_list
				class_labels_fname = "data/"+ dataset + ".Labels"
				embedding_fname = trainer.output_fh
				classify_scores = cross_val_accuracy(corpus_dir, trainer.corpus.extension, embedding_fname, class_labels_fname)
				mean_acc, std_dev = classify_scores
				print("Mean accuracy using 10 cross fold accuracy: %s with std %s" % (
				    mean_acc, std_dev))