from gensim.models import Word2Vec

import os
import time
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus
from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer

# DGK-WL
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
corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
extension = ".wld" + str(wl_depth) # Extension of the graph document

############
# Step 2
# Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
############
memory_times = []
for _ in range(10):
	start = time.time()
	model = Word2Vec(corpus, size=32, window=10, min_count=0, sg=1, hs=0, iter=100, negative=10, batch_words=128)
	end = time.time()
	end_time = end-start
	memory_times.append(end_time)

mean_mem_time = np.mean(memory_times)
std_mem_time = np.std(memory_times)
print("In Memory Geo2DR DGK-WL mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))


# DGK-SP
import os
import time
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.shortest_path_patterns import sp_corpus
from geometric2dr.embedding_methods.skipgram_trainer import InMemoryTrainer

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths for subgraph embeddings
output_embedding_fh = "SPP_Subgraph_Embeddings.json"


############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir) # will produce .spp files
extension = ".spp"
############
# Step 2
# Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
############
memory_times = []
for _ in range(10):
	start = time.time()
	model = Word2Vec(corpus, size=32, window=10, min_count=0, sg=1, hs=0, iter=100, negative=10, batch_words=128)
	end = time.time()
	end_time = end-start
	memory_times.append(end_time)

mean_mem_time = np.mean(memory_times)
std_mem_time = np.std(memory_times)
print("In Memory Geo2DR DGK-SP mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))




# DGK-GK
import os
import time
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.graphlet_patterns import graphlet_corpus
from geometric2dr.embedding_methods.skipgram_trainer import Trainer, InMemoryTrainer

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths for subgraph embeddings
output_embedding_fh = "Graphlet_Subgraph_Embeddings.json"

# Graphlet decomposition hyperparameters
num_graphlet = 7 # size of the graphlets to extract
sample_size = 100 # number of graphlets samples to extract

############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_data_dir, num_graphlet, sample_size)
extension = ".graphlet_ng_"+str(num_graphlet)+"_ss_"+str(sample_size)

############
# Step 2
# Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
############
memory_times = []
for _ in range(10):
	start = time.time()
	model = Word2Vec(corpus, size=32, window=10, min_count=0, sg=1, hs=0, iter=100, negative=10, batch_words=128)
	end = time.time()
	end_time = end-start
	memory_times.append(end_time)

mean_mem_time = np.mean(memory_times)
std_mem_time = np.std(memory_times)
print("In Memory Geo2DR DGK-GRAPHLET mean time: %.4f standard dev: %.4f " % (mean_mem_time, std_mem_time))
