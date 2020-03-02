"""
A script which times the training time of our graph2vec
reimplementations using both disk memory dataset loaders
and ram memory dataset loaders.
"""

import os
import time
import numpy as np

from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
from geometric2dr.embedding_methods.pvdbow_trainer import Trainer, InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy
import geometric2dr.embedding_methods.utils as utils

# # Input data paths
# dataset = "MUTAG"
# corpus_data_dir = "data/" + dataset

# wl_depth = 2
# min_count = 0


# emb_dimension = 128
# batch_size = 1024
# epochs = 100
# initial_lr = 0.1


# # Learn embeddings
# graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
# wl_corpus(graph_files, wl_depth)
# extension = ".wld" + str(wl_depth) # Extension of the graph document
# output_embedding_fh = "runtime_analysis_embeddings"


# # Load from disk trainer
# hd_times = []
# for _ in range(10):
# 	trainer = Trainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=output_embedding_fh,
# 	                  emb_dimension=emb_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
# 	                  min_count=min_count)
# 	start_time = time.time()
# 	trainer.train()
# 	end_time = (time.time() - start_time)
# 	hd_times.append(end_time)

# mean_hd_time = np.mean(hd_times)
# std_hd_time = np.std(hd_times)


# # Use memory trainer
# memory_times = []
# for _ in range(10):
# 	trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=output_embedding_fh,
# 	                  emb_dimension=emb_dimension, batch_size=batch_size, epochs=epochs, initial_lr=initial_lr,
# 	                  min_count=min_count)
	# start_time = time.time()
	# trainer.train()
	# end_time = (time.time() - start_time)
	# memory_times.append(end_time)

# mean_mem_time = np.mean(memory_times)
# std_mem_time = np.std(memory_times)


# # print("Hard Drive Geo2DR Graph2Vec mean time: %.4f standard dev: %.4f " % (mean_hd_time, std_hd_time))
# print("In Memory Geo2DR Graph2Vec mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))



# Anonymous Walk Embeddings
import os
import time
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.anonymous_walk_patterns import awe_corpus
from geometric2dr.embedding_methods.classify import cross_val_accuracy
from geometric2dr.embedding_methods.pvdm_trainer import PVDM_Trainer # Note use of PVDM

aw_length = 10
label_setting = "nodes" # AWE is quite nice and versatile allowing for different node-label/edge-label settings

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths
output_embedding_fh = "AWE_Embeddings.json"

#######
# Step 1 Create corpus data for neural language model
# We keep permanent files for sake of deeper post studies and testing
#######
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)

memory_times = []
for _ in range(10):
	awe_corpus(corpus_data_dir, aw_length, label_setting, saving_graph_docs=True)
	extension = ".awe_" + str(aw_length) + "_" + label_setting

	######
	# Step 2 Train a neural language model to learn distributed representations
	# 		 of the graphs directly or of its substructures. Here we learn it directly
	#		 for an example of the latter check out the DGK models.
	######
	trainer = PVDM_Trainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=16, output_fh=output_embedding_fh,
	                  emb_dimension=128, batch_size=100, epochs=100, initial_lr=0.1, min_count=0)
	start_time = time.time()
	trainer.train()
	end_time = (time.time() - start_time)
	memory_times.append(end_time)

mean_mem_time = np.mean(memory_times)
std_mem_time = np.std(memory_times)
print("In Memory Geo2DR AWE-DD mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))




# import os
# import time
# import numpy as np

# import geometric2dr.embedding_methods.utils as utils
# from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
# from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer

# # DGK-WL
# # Input data paths
# dataset = "MUTAG"
# corpus_data_dir = "data/" + dataset

# # Desired output paths for subgraph embeddings
# output_embedding_fh = "WL_Subgraph_Embeddings.json"

# # WL decomposition hyperparameters
# wl_depth = 2

# ############
# # Step 1
# # Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
# ############
# graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
# corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
# extension = ".wld" + str(wl_depth) # Extension of the graph document

# ############
# # Step 2
# # Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
# ############
# memory_times = []
# for _ in range(5):
# 	trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh=output_embedding_fh,
# 					  emb_dimension=32, batch_size=1280, epochs=100, initial_lr=0.1, min_count=1)
# 	start_time = time.time()
# 	trainer.train()
# 	end_time = (time.time() - start_time)
# 	memory_times.append(end_time)

# mean_mem_time = np.mean(memory_times)
# std_mem_time = np.std(memory_times)
# print("In Memory Geo2DR DGK-WL mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))


# # DGK-SP
# import os
# import time
# import numpy as np

# import geometric2dr.embedding_methods.utils as utils
# from geometric2dr.decomposition.shortest_path_patterns import sp_corpus
# from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer

# # Input data paths
# dataset = "MUTAG"
# corpus_data_dir = "data/" + dataset

# # Desired output paths for subgraph embeddings
# output_embedding_fh = "SPP_Subgraph_Embeddings.json"


# ############
# # Step 1
# # Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
# ############
# graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
# corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir) # will produce .spp files
# extension = ".spp"
# ############
# # Step 2
# # Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
# ############
# memory_times = []
# for _ in range(5):
# 	trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh=output_embedding_fh,
# 				  emb_dimension=32, batch_size=128, epochs=100, initial_lr=0.1,
# 				  min_count=1)
# 	start_time = time.time()
# 	trainer.train()
# 	end_time = (time.time() - start_time)
# 	memory_times.append(end_time)

# mean_mem_time = np.mean(memory_times)
# std_mem_time = np.std(memory_times)
# print("In Memory Geo2DR DGK-SP mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))




# # # DGK-GK
# import os
# import time
# import numpy as np

# import geometric2dr.embedding_methods.utils as utils
# from geometric2dr.decomposition.graphlet_patterns import graphlet_corpus
# from geometric2dr.embedding_methods.skipgram_trainer import Trainer, InMemoryTrainer

# # Input data paths
# dataset = "MUTAG"
# corpus_data_dir = "data/" + dataset

# # Desired output paths for subgraph embeddings
# output_embedding_fh = "Graphlet_Subgraph_Embeddings.json"

# # Graphlet decomposition hyperparameters
# num_graphlet = 7 # size of the graphlets to extract
# sample_size = 100 # number of graphlets samples to extract

# ############
# # Step 1
# # Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
# ############
# graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
# corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_data_dir, num_graphlet, sample_size)
# extension = ".graphlet_ng_"+str(num_graphlet)+"_ss_"+str(sample_size)

# ############
# # Step 2
# # Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
# ############
# memory_times = []
# for _ in range(5):
# 	trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, window_size=10, output_fh=output_embedding_fh,
# 					  emb_dimension=32, batch_size=128, epochs=100, initial_lr=0.1,
# 					  min_count=0)
# 	start_time = time.time()
# 	trainer.train()
# 	end_time = (time.time() - start_time)
# 	memory_times.append(end_time)

# mean_mem_time = np.mean(memory_times)
# std_mem_time = np.std(memory_times)
# print("In Memory Geo2DR DGK-GRAPHLET mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))
