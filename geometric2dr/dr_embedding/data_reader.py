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
		