"""
Experiment script which runs sets up and builds distributed representations of graphs using 
the WL decomposition algorithm at different degrees.

Then we build the distributed representations using the skipgram with different:
- embedding_dimensions
- batch_size
- epochs
- initial_learning rate (fixed for now)
- min_count (on number of subgraph patterns)

Author: Paul Scherer
"""
import embedding_methods.utils as utils
from embedding_methods.classify import cross_val_accuracy
from embedding_methods.trainer import Trainer

# wl_degrees = [1,2,3]
# embedding_dimensions = [8, 16, 32, 64, 128]
# batch_size = 128
# runs = 5

# dataset = "MUTAG"
# corpus_dir = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/" + dataset

# for wl_h in wl_degrees:
# 	extension = ".wld" + str(wl_h)
# 	for embedding_dimension in embedding_dimensions:
# 		for run in range(runs):
# 			# Set the output file name for the embeddings
# 			output_fh = str.join("_", [dataset, "WL", str(wl_h), "ed", str(embedding_dimension), "bs", str(batch_size)])
# 			output_fh += ".json"

# 			# Run the decomposition algorithm
# 			graph_files = utils.get_files(corpus_dir, ".gexf", max_files=0)
# 			print("###.... Loaded %s files in total" % (str(len(graph_files))))
# 			# wlk_relabeled_corpus(graph_files, wl_h)
