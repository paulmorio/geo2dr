"""
DataReader class

Is essentially the class we use to handle the data and corpi
"""

import numpy as np
import torch

from torch.utils.data import Dataset

np.random.seed(27)

class DataReader(object):
	"""docstring for DataReader"""
	def __init__(self, arg):
		super(DataReader, self).__init__()
		self.arg = arg
		
class GraphCorpus(Dataset):
	"""
	A DataSet extension that handles graph corpuses, ie graphs and their 
	induced *contexts* subgraphs
	"""

	def __init__(self, data, context="all_induced"):
		self.data = data
		self.context = context


	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass

	@staticmethod
	def collate(batches):
		all_targets = [target for batch in batches for target, _, _ in batch if len(batch)>0]
		all_contexts = [context for batch in batches for _, context, _ in batch if len(batch)>0]
		all_neg_contexts = [neg_context for batch in batches for _, _, neg_context in batch if len(batch)>0]