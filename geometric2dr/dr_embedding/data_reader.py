"""
DataReader class

Is essentially the class we use to handle the data and corpi
"""

import numpy as np
import torch

from torch.utils.data import Dataset

np.random.seed(27)

class DataReader(object):
	NEGATIVE_TABLE_SIZE=1e8

	def __init__(self, input_graph_folder, min_count):
		self.negatives = []
		self.discards = []
		self.negpos = 0

		self.graph2id = dict()
		self.id2graph = dict()
		self.sentences_count = 0
		self.token_count = 0
		self.word_frequency = dict()

		self.input_graph_folder = input_graph_folder # path containing all of the GEXF and .WLDR Files
		self.read_graphs(min_count)
		self.initTableNegatives()
		self.initTableDiscards()

	def read_graphs(self, min_count):
		"""
		Our equivalent of read_words, except we also have to be careful of seperating graphID with subgraphID
		"""
		print("Total graph embeddings: " + str(len(self.graph2id)))
		print("Total subgraphs and subgraph embeddings" + str(subgraphID))
		pass

	# Needs understanding and translations but line complete
	def initTableDiscards(self):
		t = 1e-4
		f = np.array(list(self.word_frequency.values())) / self.token_count
		self.discards = np.sqrt(t/f)+(t/f)

	# Needs translations but line complete
	def initTableNegatives(self):
		pow_frequency = np.array(list(self.word_frequency.values()))**0.5
		words_pow = sum(pow_frequency)
		ratio = pow_frequency / words_pow
		count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
		for wid, c in enumerate(count):
			self.negatives += [wid]*int(c)
		self.negatives = np.array(self.negatives)
		np.random.shuffle(self.negatives)

	# can stay as is line complete.
	def getNegatives(self, target, size): # check equality with target
		response = self.negatives[self.negpos:self.negpos + size]
		self.negpos = (self.negpos + size) % len(self.negatives)
		if len(response) != size:
			return np.concatenate((response, self.negatives[0:self.negpos]))
		return response

class GraphCorpusSkipgram(Dataset):
	"""
	A DataSet extension that handles graph corpuses, ie graphs and their 
	induced *contexts* subgraphs
	"""

	def __init__(self, data, context="all_induced"):
		self.data = data
		self.context = context
		self.input_graph_folder

	def __len__(self):
		# Return the number of graphs
		pass

	def __getitem__(self, idx):
		# Return a tuple of graphids, the context subgraphs, and negative samples: a training input (target, context, negative)
		# for the learning algorithm
		pass

	@staticmethod
	def collate(batches):
		all_targets = [target for batch in batches for target, _, _ in batch if len(batch)>0]
		all_contexts = [context for batch in batches for _, context, _ in batch if len(batch)>0]
		all_neg_contexts = [neg_context for batch in batches for _, _, neg_context in batch if len(batch)>0]

		return torch.LongTensor(all_targets), torch.LongTensor(all_contexts), torch.LongTensor(all_neg_contexts)