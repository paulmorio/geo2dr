"""
An example reimplementation of DGK-Graphlets (Yanardag and Vishwanathan 2015) 
using Geo2DR (geometric2dr) and Gensim to show compatibility

Author: Paul Scherer 2020
"""
import numpy as np

import geometric2dr.embedding_methods.utils as utils
from geometric2dr.decomposition.graphlet_patterns import graphlet_corpus
from geometric2dr.embedding_methods.skipgram_trainer import Trainer, InMemoryTrainer
from geometric2dr.embedding_methods.classify import cross_val_accuracy_rbf_bag_of_words

from gensim.models import Word2Vec

# Input data paths
dataset = "MUTAG"
corpus_data_dir = "data/" + dataset

# Desired output paths for subgraph embeddings
output_embedding_fh = "Graphlet_Subgraph_Embeddings.json"

# Graphlet decomposition hyperparameters
num_graphlet = 8 # size of the graphlets to extract
sample_size = 50 # number of graphlets samples to extract


############
# Step 1
# Run the decomposition algorithm to get subgraph patterns across the graphs of MUTAG
############
graph_files = utils.get_files(corpus_data_dir, ".gexf", max_files=0)
corpus, vocabulary, prob_map, num_graphs, graph_map = graphlet_corpus(corpus_data_dir, num_graphlet, sample_size)
extension = ".graphlet_ng_"+str(num_graphlet)+"_ss_"+str(sample_size)

corpus = [[str(x) for x in group] for group in corpus] # as gensim doesn't like integers

############
# Step 2
# Train a skipgram (w. Negative Sampling) model to learn distributed representations of the subgraph patterns
############
model = Word2Vec(corpus, size=32, window=10, min_count=0, sg=1, hs=0, iter=5, negative=10, batch_words=128)

############
# Step 3
# Create a kernel matrix of the graphs using the embeddings of the substructures
############
def l2_norm(vec):
	return  np.sqrt(np.dot(vec, vec))

# deep substructure embeddings w/l2 norm
P = np.zeros((num_graphs, len(vocabulary)))
for i in range(num_graphs):
	for jdx, j in enumerate(vocabulary):
		P[i][jdx] = prob_map[i+1].get(j,0)
M = np.zeros((len(vocabulary), len(vocabulary)))
for idx,i in enumerate(vocabulary):
	M[idx][idx] = l2_norm(model[str(i)])
S = (P.dot(M))

###########
# Step 4
# Use some kernel method, such as an SVM to compute classifications on the graph kernel matrix
###########
class_labels_fname = "data/"+ dataset + ".Labels"
xylabels = utils.get_class_labels_tuples(graph_files,class_labels_fname)
xylabels.sort(key=lambda tup: tup[0])
kernel_row_x_id, kernel_row_y_id = zip(*xylabels)

acc, std = cross_val_accuracy_rbf_bag_of_words(S, kernel_row_y_id)
print ('#... Accuracy score: %0.4f, Standard deviation: %0.4f' % (acc, std))