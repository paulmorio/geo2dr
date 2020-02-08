"""
DataReader class

Is essentially the class we use to handle the data and corpi
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from random import shuffle

from utils import get_files

np.random.seed(27)
class Corpus(Dataset):
	"""
	Class which representes all of the graph documents in a graph dataset
	"""
	NEGATIVE_TABLE_SIZE=1e8

	def __init__(self, corpus_dir=None, extension=".wld2", max_files=0, min_count=0):
		assert corpus_dir != None, "Please specify the path where the graph files are"
		self.corpus_dir = corpus_dir
		self.extension = extension
		self.graph_index = 0
		self.subgraph_index = 0
		self.epoch_flag = 0
		self.max_files = max_files
		self.graph_ids_for_batch_traversal = []
		self.min_count = min_count

		self.negatives = []
		self.discards = []
		self.negpos = 0

		# Methods
		self.scan_and_load_corpus()
		self.initTableDiscards()
		self.initTableNegatives()

	def scan_and_load_corpus(self):
		"""
		gets the list of graph file paths, gives them number ids in a map and calls 
		scan_corpus also makes available a list of shuffled graph_ids for batch
		"""
		print("Scanning and Loading Corpus")
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)
		self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)}
		self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}

		# Scan the corpus
		# This creates the alphabet and vocabulary of subgraphs
		subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

		print("Setting batch travel ids")
		self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
		shuffle(self.graph_ids_for_batch_traversal)

	def scan_corpus(self, min_count):
		"""
		Maps the graph files to a subgraph alphabet from which we create new_ids for the subgraphs
		which in turn get used by the skipgram architectures
		"""
		# Get all the subgraph names (the centers, ie without context around itself) ie first item of 
		# each line in the grapdoc files
		subgraphs = []
		for fname in self.graph_fname_list:
			subgraphs.extend([l.split()[0] for l in open(fname).readlines()])
		subgraphs.append('UNK') # end flag

		# Frequency map of subgraph types
		subgraph_to_freq_map = Counter(subgraphs)
		del subgraphs

		# Remove infrequent graph patterns if user specified a min_count
		if min_count:
			subgraph_to_freq_map = dict(subgraph_to_freq_map)
			for key in list(subgraph_to_freq_map.keys()):
				if subgraph_to_freq_map[key] < min_count:
					subgraph_to_freq_map.pop(key, None)

		# Also give each of the subgraph labels new int ids
		subgraph_to_id_map = {sg:i for i, sg, in enumerate(subgraph_to_freq_map.keys())}

		self._subgraph_to_freq_map = subgraph_to_freq_map
		self._subgraph_to_id_map = subgraph_to_id_map
		self._id_to_subgraph_map = {v:k  for k,v in subgraph_to_id_map.items()}
		self._subgraphcount = sum(subgraph_to_freq_map.values()) # token count, the number of subgraphs in all graphs


		self.num_graphs = len(self.graph_fname_list) # number of targets
		self.num_subgraphs = len(subgraph_to_id_map) # size of vocabulary

		# This is a list sorted by id, which contains the frequency of a subgraph appearing as a list
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])

		return self._subgraph_to_id_map

	def add_file(self, full_graph_path):
		"""
		This method is used to add new graphs into the corpus for inductive learning of new unseen graphs
		"""

		# Retrieve the graphs files and assign them internal ids for this method
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)

		if full_graph_path in self.graph_fname_list:
			# if the graph is already in the corpus we may ignore it
			print("The graph %s is already in the corpus" % (full_graph_path))
			return
		else:
			self.graph_fname_list.append(full_graph_path) # add the new file
			self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)} # add to the end
			self._graph_name_to_id_map[full_graph_path] = self.num_graphs
			self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}
				
			# Scan the corpus (ie explain in more detail in a sec)
			# This creates an "alphabet" and "vocabulary" of subgraphs
			subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

			self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
			shuffle(self.graph_ids_for_batch_traversal)

			self.initTableDiscards()
			self.initTableNegatives()

	# should run this as part of init
	def initTableDiscards(self):
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])
		self.subgraph_id_freq_map = {}
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map[i] = self.subgraph_id_freq_map_as_list[i]

		t = 1e-4
		f = np.array(list(self.subgraph_id_freq_map.values())) / self._subgraphcount
		self.discards = np.sqrt(t/f)+(t/f)

	# Needs translations but line complete
	def initTableNegatives(self):
		pow_frequency = np.array(list(self.subgraph_id_freq_map.values()))**0.5
		words_pow = sum(pow_frequency)
		ratio = pow_frequency / words_pow
		count = np.round(ratio * Corpus.NEGATIVE_TABLE_SIZE)
		for sg_id, c in enumerate(count):
			self.negatives += [sg_id]*int(c)
		self.negatives = np.array(self.negatives)
		np.random.shuffle(self.negatives)

	def getNegatives(self, target, size): 
		response = self.negatives[self.negpos:self.negpos + size]
		self.negpos = (self.negpos + size) % len(self.negatives)
		if target in response: # check equality with target
			for i in np.where(response == target):
				response[i] = np.random.randint(0,self.num_subgraphs)
		if len(response) != size:
			return np.concatenate((response, self.negatives[0:self.negpos]))
		return response

	def __len__(self):
		# Return the number of total number of subgraphs
		# print("The number of graph-subgraph pairs considering batchsize in the dataset is")
		return self._subgraphcount

	def __getitem__(self, idx):
		# Get a single item of data from the dataset.
		target_graph_ids = []
		context_subgraph_ids = []

		# Extract a random graph and read its contents
		graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]] # pull out a random graph (its filename here)
		graph_contents = open(graph_name).readlines()
		
		# tldr: random graph traverser 
		# If we've looked at all the subgraphs we go to the next graph
		# We set the epoch flag true if we've also gone through n graphs (in a n graph dataset)
		# we then grab the next graph file whether that be the next graph in the shuffled list
		# or the first graph in a reshuffled list of graph files
		while self.subgraph_index >= len(graph_contents):
			self.subgraph_index = 0
			self.graph_index += 1
			if self.graph_index == len(self.graph_fname_list):
				self.graph_index = 0
				np.random.shuffle(self.graph_ids_for_batch_traversal)
				self.epoch_flag = True
			graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
			graph_contents = open(graph_name).readlines()

		# Given that we haven't gotten enough graphs for our batch
		# We traverse the file at graph_name and graph the center as the (context subgraph)
		# which is a bit counter-intuitive but we consider the centers as the "context" of the
		# graph as a whole.
		# while len(context_subgraph_ids) < batch_size:
		while len(context_subgraph_ids) < 1:
			line_id = self.subgraph_index
			context_subgraph = graph_contents[line_id].split()[0] # first item on the line
			target_graph = graph_name

			if context_subgraph in self._subgraph_to_id_map:
				context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
				target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

			# context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
			# target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

			# move on to the next subgraph
			self.subgraph_index += 1

			# if we've reached the end of the graph file (ie exhasuted all the subgraphs in it)
			# we move onto the next graph as above.
			while self.subgraph_index == len(graph_contents):
				self.subgraph_index = 0
				self.graph_index +=1 
				if self.graph_index == len(self.graph_fname_list):
					self.graph_index = 0
					np.random.shuffle(self.graph_ids_for_batch_traversal)
					self.epoch_flag = True

				graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
				graph_contents = open(graph_name).readlines()
		
		# Once we've built 'batch_size' number worth of targets and contexts
		# we zip them, shuffle them, and unzip the pairs in shuffled order (keeping the pairing)
		target_context_pairs = list(zip(target_graph_ids, context_subgraph_ids))
		shuffle(target_context_pairs)
		target_graph_ids, context_subgraph_ids = list(zip(*target_context_pairs))

		negatives_per_context = [self.getNegatives(x,10) for x in context_subgraph_ids]
		target_context_negatives = [(target, context, negatives) for (target, context, negatives) in zip(target_graph_ids, context_subgraph_ids, negatives_per_context)]
		return target_context_negatives

	@staticmethod
	def collate(batches):
		all_targets = [target for batch in batches for target, _, _ in batch if len(batch)>0]
		all_contexts = [context for batch in batches for _, context, _ in batch if len(batch)>0]
		all_neg_contexts = [neg_context for batch in batches for _, _, neg_context in batch if len(batch)>0]

		return torch.LongTensor(all_targets), torch.LongTensor(all_contexts), torch.LongTensor(all_neg_contexts)









class InMemoryCorpus(Dataset):
	"""
	Class which representes all of the graph documents in a graph dataset, This version keeps the entire corpus with negatives in memory
	which requires a larger initial creation time but is much quicker
	"""
	NEGATIVE_TABLE_SIZE=1e8

	def __init__(self, corpus_dir=None, extension=".wld2", max_files=0, min_count=0):
		assert corpus_dir != None, "Please specify the path where the graph files are"
		self.corpus_dir = corpus_dir
		self.extension = extension
		self.graph_index = 0
		self.subgraph_index = 0
		self.epoch_flag = 0
		self.max_files = max_files
		self.graph_ids_for_batch_traversal = []
		self.min_count = min_count

		self.negatives = []
		self.discards = []
		self.negpos = 0

		# Methods
		self.scan_and_load_corpus()
		self.initTableDiscards()
		self.initTableNegatives()
		self.pre_load_corpus()

	def scan_and_load_corpus(self):
		"""
		gets the list of graph file paths, gives them number ids in a map and calls 
		scan_corpus also makes available a list of shuffled graph_ids for batch
		"""
		print("Scanning and Loading Corpus")
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)
		self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)}
		self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}

		# Scan the corpus
		# This creates the alphabet and vocabulary of subgraphs
		subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

		print("Setting batch travel ids")
		self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
		shuffle(self.graph_ids_for_batch_traversal)

	def scan_corpus(self, min_count):
		"""
		Maps the graph files to a subgraph alphabet from which we create new_ids for the subgraphs
		which in turn get used by the skipgram architectures
		"""
		# Get all the subgraph names (the centers, ie without context around itself) ie first item of 
		# each line in the grapdoc files
		subgraphs = []
		for fname in self.graph_fname_list:
			subgraphs.extend([l.split()[0] for l in open(fname).readlines()])
		subgraphs.append('UNK') # end flag

		# Frequency map of subgraph types
		subgraph_to_freq_map = Counter(subgraphs)
		del subgraphs

		# Remove infrequent graph patterns if user specified a min_count
		if min_count:
			subgraph_to_freq_map = dict(subgraph_to_freq_map)
			for key in list(subgraph_to_freq_map.keys()):
				if subgraph_to_freq_map[key] < min_count:
					subgraph_to_freq_map.pop(key, None)

		# Also give each of the subgraph labels new int ids
		subgraph_to_id_map = {sg:i for i, sg, in enumerate(subgraph_to_freq_map.keys())}

		self._subgraph_to_freq_map = subgraph_to_freq_map
		self._subgraph_to_id_map = subgraph_to_id_map
		self._id_to_subgraph_map = {v:k  for k,v in subgraph_to_id_map.items()}
		self._subgraphcount = sum(subgraph_to_freq_map.values()) # token count, the number of subgraphs in all graphs


		self.num_graphs = len(self.graph_fname_list) # number of targets
		self.num_subgraphs = len(subgraph_to_id_map) # size of vocabulary

		# This is a list sorted by id, which contains the frequency of a subgraph appearing as a list
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])

		return self._subgraph_to_id_map

	def add_file(self, full_graph_path):
		"""
		This method is used to add new graphs into the corpus for inductive learning of new unseen graphs
		"""

		# Retrieve the graphs files and assign them internal ids for this method
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)

		if full_graph_path in self.graph_fname_list:
			# if the graph is already in the corpus we may ignore it
			print("The graph %s is already in the corpus" % (full_graph_path))
			return
		else:
			self.graph_fname_list.append(full_graph_path) # add the new file
			self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)} # add to the end
			self._graph_name_to_id_map[full_graph_path] = self.num_graphs
			self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}
				
			# Scan the corpus (ie explain in more detail in a sec)
			# This creates an "alphabet" and "vocabulary" of subgraphs
			subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

			self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
			shuffle(self.graph_ids_for_batch_traversal)

			self.initTableDiscards()
			self.initTableNegatives()

	# should run this as part of init
	def initTableDiscards(self):
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])
		self.subgraph_id_freq_map = {}
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map[i] = self.subgraph_id_freq_map_as_list[i]

		t = 1e-4
		f = np.array(list(self.subgraph_id_freq_map.values())) / self._subgraphcount
		self.discards = np.sqrt(t/f)+(t/f)

	# Needs translations but line complete
	def initTableNegatives(self):
		pow_frequency = np.array(list(self.subgraph_id_freq_map.values()))**0.5
		words_pow = sum(pow_frequency)
		ratio = pow_frequency / words_pow
		count = np.round(ratio * Corpus.NEGATIVE_TABLE_SIZE)
		for sg_id, c in enumerate(count):
			self.negatives += [sg_id]*int(c)
		self.negatives = np.array(self.negatives)
		np.random.shuffle(self.negatives)

	def getNegatives(self, target, size): 
		response = self.negatives[self.negpos:self.negpos + size]
		self.negpos = (self.negpos + size) % len(self.negatives)
		if target in response: # check equality with target
			for i in np.where(response == target):
				response[i] = np.random.randint(0,self.num_subgraphs)
		if len(response) != size:
			return np.concatenate((response, self.negatives[0:self.negpos]))
		return response


	def pre_load_corpus(self):
		"""
		Loads an entire context-pair dataset into memory
		"""
		print("#... Generating dataset in memory for quick dataloader access")
		# Get a single item of data from the dataset.
		self.context_pair_dataset = []

		while self.epoch_flag == False:

			target_graph_ids = []
			context_subgraph_ids = []

			# Extract a random graph and read its contents
			graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]] # pull out a random graph (its filename here)
			graph_contents = open(graph_name).readlines()
			
			# tldr: random graph traverser 
			# If we've looked at all the subgraphs we go to the next graph
			# We set the epoch flag true if we've also gone through n graphs (in a n graph dataset)
			# we then grab the next graph file whether that be the next graph in the shuffled list
			# or the first graph in a reshuffled list of graph files
			while self.subgraph_index >= len(graph_contents):
				self.subgraph_index = 0
				self.graph_index += 1
				if self.graph_index == len(self.graph_fname_list):
					self.graph_index = 0
					np.random.shuffle(self.graph_ids_for_batch_traversal)
					self.epoch_flag = True
				graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
				graph_contents = open(graph_name).readlines()

			# Given that we haven't gotten enough graphs for our batch
			# We traverse the file at graph_name and graph the center as the (context subgraph)
			# which is a bit counter-intuitive but we consider the centers as the "context" of the
			# graph as a whole.
			# while len(context_subgraph_ids) < batch_size:
			# while len(context_subgraph_ids) < 1:
			line_id = self.subgraph_index
			context_subgraph = graph_contents[line_id].split()[0] # first item on the line
			target_graph = graph_name

			if context_subgraph in self._subgraph_to_id_map:
				context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
				target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

			# context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
			# target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

			# move on to the next subgraph
			self.subgraph_index += 1

			# if we've reached the end of the graph file (ie exhasuted all the subgraphs in it)
			# we move onto the next graph as above.
			while self.subgraph_index == len(graph_contents):
				self.subgraph_index = 0
				self.graph_index +=1 
				if self.graph_index == len(self.graph_fname_list):
					self.graph_index = 0
					np.random.shuffle(self.graph_ids_for_batch_traversal)
					self.epoch_flag = True

				graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
				graph_contents = open(graph_name).readlines()
			
			# Once we've built 'batch_size' number worth of targets and contexts
			# we zip them, shuffle them, and unzip the pairs in shuffled order (keeping the pairing)
			target_context_pairs = list(zip(target_graph_ids, context_subgraph_ids))
			shuffle(target_context_pairs)
			target_graph_ids, context_subgraph_ids = list(zip(*target_context_pairs))

			negatives_per_context = [self.getNegatives(x,10) for x in context_subgraph_ids]
			target_context_negatives = [(target, context, negatives) for (target, context, negatives) in zip(target_graph_ids, context_subgraph_ids, negatives_per_context)]
			self.context_pair_dataset.append(target_context_negatives)
		self.epoch_flag = False
		self.dataset_traverser = 0

	def __len__(self):
		# Return the number of total number of subgraphs
		# print("The number of graph-subgraph pairs considering batchsize in the dataset is")
		return self._subgraphcount

	def __getitem__(self, idx):
		if self.dataset_traverser == len(self.graph_fname_list):
			self.dataset_traverser = 0
			target_context_negatives = self.context_pair_dataset[self.dataset_traverser]
			self.dataset_traverser += 1
			return target_context_negatives
		else:
			target_context_negatives = self.context_pair_dataset[self.dataset_traverser]
			self.dataset_traverser += 1
			return target_context_negatives

	@staticmethod
	def collate(batches):
		all_targets = [target for batch in batches for target, _, _ in batch if len(batch)>0]
		all_contexts = [context for batch in batches for _, context, _ in batch if len(batch)>0]
		all_neg_contexts = [neg_context for batch in batches for _, _, neg_context in batch if len(batch)>0]

		return torch.LongTensor(all_targets), torch.LongTensor(all_contexts), torch.LongTensor(all_neg_contexts)



class PVDMCorpus(Dataset):
	"""
	Class which representes all of the graph documents in a graph dataset serves context for PVDM models, This version 
	keeps the entire corpus with negatives in memory which requires a larger initial creation time but is much quicker
	"""
	NEGATIVE_TABLE_SIZE=1e8

	def __init__(self, corpus_dir=None, extension=".wld2", max_files=0, min_count=0, window_size=1):
		assert corpus_dir != None, "Please specify the path where the graph files are"
		assert window_size > 0, "Please make a window size that is greater than 0"
		self.corpus_dir = corpus_dir
		self.extension = extension
		self.graph_index = 0
		self.subgraph_index = 0
		self.epoch_flag = 0
		self.max_files = max_files
		self.graph_ids_for_batch_traversal = []
		self.min_count = min_count
		self.window_size = window_size

		self.negatives = []
		self.discards = []
		self.negpos = 0

		# Methods
		self.scan_and_load_corpus()
		self.initTableDiscards()
		self.initTableNegatives()
		self.pre_load_corpus()

	def scan_and_load_corpus(self):
		"""
		gets the list of graph file paths, gives them number ids in a map and calls 
		scan_corpus also makes available a list of shuffled graph_ids for batch
		"""
		print("Scanning and Loading Corpus")
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)
		self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)}
		self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}

		# Scan the corpus
		# This creates the alphabet and vocabulary of subgraphs
		subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

		print("Setting batch travel ids")
		self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
		shuffle(self.graph_ids_for_batch_traversal)

	def scan_corpus(self, min_count):
		"""
		Maps the graph files to a subgraph alphabet from which we create new_ids for the subgraphs
		which in turn get used by the skipgram architectures
		"""
		# Get all the subgraph names (the centers, ie without context around itself) ie first item of 
		# each line in the grapdoc files
		subgraphs = defaultdict(int)
		for fname in self.graph_fname_list:
			for l in open(fname).readlines():
				ss = l.split()[:self.window_size+1] # as context may have new subgraphs we havent seen				
				for s in ss:
					subgraphs[s] += 1


		# Frequency map of subgraph types
		subgraph_to_freq_map = subgraphs

		# Remove infrequent graph patterns if user specified a min_count
		if min_count:
			subgraph_to_freq_map = dict(subgraph_to_freq_map)
			for key in list(subgraph_to_freq_map.keys()):
				if subgraph_to_freq_map[key] < min_count:
					subgraph_to_freq_map.pop(key, None)

		# Also give each of the subgraph labels new int ids
		subgraph_to_id_map = {sg:i for i, sg, in enumerate(subgraph_to_freq_map.keys())}

		self._subgraph_to_freq_map = subgraph_to_freq_map
		self._subgraph_to_id_map = subgraph_to_id_map
		self._id_to_subgraph_map = {v:k  for k,v in subgraph_to_id_map.items()}
		self._subgraphcount = sum(subgraph_to_freq_map.values()) # token count, the number of subgraphs in all graphs


		self.num_graphs = len(self.graph_fname_list) # number of targets
		self.num_subgraphs = len(subgraph_to_id_map) # size of vocabulary

		# This is a list sorted by id, which contains the frequency of a subgraph appearing as a list
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])

		return self._subgraph_to_id_map

	def add_file(self, full_graph_path):
		"""
		This method is used to add new graphs into the corpus for inductive learning of new unseen graphs
		"""

		# Retrieve the graphs files and assign them internal ids for this method
		self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)

		if full_graph_path in self.graph_fname_list:
			# if the graph is already in the corpus we may ignore it
			print("The graph %s is already in the corpus" % (full_graph_path))
			return
		else:
			self.graph_fname_list.append(full_graph_path) # add the new file
			self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)} # add to the end
			self._graph_name_to_id_map[full_graph_path] = self.num_graphs
			self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}
				
			# Scan the corpus (ie explain in more detail in a sec)
			# This creates an "alphabet" and "vocabulary" of subgraphs
			subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

			self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
			shuffle(self.graph_ids_for_batch_traversal)

			self.initTableDiscards()
			self.initTableNegatives()

	# should run this as part of init
	def initTableDiscards(self):
		self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])
		self.subgraph_id_freq_map = {}
		for i in range(len(self._subgraph_to_freq_map)):
			self.subgraph_id_freq_map[i] = self.subgraph_id_freq_map_as_list[i]

		t = 1e-4
		f = np.array(list(self.subgraph_id_freq_map.values())) / self._subgraphcount
		self.discards = np.sqrt(t/f)+(t/f)

	def initTableNegatives(self):
		pow_frequency = np.array(list(self.subgraph_id_freq_map.values()))**0.5
		words_pow = sum(pow_frequency)
		ratio = pow_frequency / words_pow
		count = np.round(ratio * Corpus.NEGATIVE_TABLE_SIZE)
		for sg_id, c in enumerate(count):
			self.negatives += [sg_id]*int(c)
		self.negatives = np.array(self.negatives)
		np.random.shuffle(self.negatives)

	def getNegatives(self, target, size): 
		response = self.negatives[self.negpos:self.negpos + size]
		self.negpos = (self.negpos + size) % len(self.negatives)
		if target in response: # check equality with target
			for i in np.where(response == target):
				response[i] = np.random.randint(0,self.num_subgraphs)
		if len(response) != size:
			return np.concatenate((response, self.negatives[0:self.negpos]))
		return response


	def pre_load_corpus(self):
		"""
		Loads an entire context-pair dataset into memory
		"""
		print("#... Generating dataset in memory for quick dataloader access")
		# Get a single item of data from the dataset.
		self.context_pair_dataset = []

		while self.epoch_flag == False:

			target_graph_ids = []
			target_context_subgraph_ids = []
			subgraph_contexts_ids = []

			# Extract a random graph and read its contents
			graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]] # pull out a random graph (its filename here)
			graph_contents = open(graph_name).readlines()
			
			# tldr: random graph traverser 
			# If we've looked at all the subgraphs we go to the next graph
			# We set the epoch flag true if we've also gone through n graphs (in a n graph dataset)
			# we then grab the next graph file whether that be the next graph in the shuffled list
			# or the first graph in a reshuffled list of graph files
			while self.subgraph_index >= len(graph_contents):
				self.subgraph_index = 0
				self.graph_index += 1
				if self.graph_index == len(self.graph_fname_list):
					self.graph_index = 0
					np.random.shuffle(self.graph_ids_for_batch_traversal)
					self.epoch_flag = True
				graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
				graph_contents = open(graph_name).readlines()

			# Given that we haven't gotten enough graphs for our batch
			# We traverse the file at graph_name and graph the center as the (context subgraph)
			# which is a bit counter-intuitive but we consider the centers as the "context" of the
			# graph as a whole.
			# while len(context_subgraph_ids) < batch_size:
			# while len(context_subgraph_ids) < 1:
			line_id = self.subgraph_index
			target_context = graph_contents[line_id].split()[0] # first item on the line is the center
			subgraph_contexts = graph_contents[line_id].split()[1:1+self.window_size] # the contexts of the center we try to predict

			target_graph = graph_name

			if target_context in self._subgraph_to_id_map:
				target_context_subgraph_ids.append(self._subgraph_to_id_map[target_context]) # add the ids of the subgraph into the context
				target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target
				temp_subgraph_contexts = []
				for subgraph_context in subgraph_contexts:
					if subgraph_context in self._subgraph_to_id_map:
						temp_subgraph_contexts.append(self._subgraph_to_id_map[subgraph_context])
				subgraph_contexts_ids.append(temp_subgraph_contexts)

			# move on to the next subgraph
			self.subgraph_index += 1

			# if we've reached the end of the graph file (ie exhasuted all the subgraphs in it)
			# we move onto the next graph as above.
			while self.subgraph_index == len(graph_contents):
				self.subgraph_index = 0
				self.graph_index +=1 
				if self.graph_index == len(self.graph_fname_list):
					self.graph_index = 0
					np.random.shuffle(self.graph_ids_for_batch_traversal)
					self.epoch_flag = True

				graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
				graph_contents = open(graph_name).readlines()
			
			# Once we've built 'batch_size' number worth of targets and contexts
			# we zip them, shuffle them, and unzip the pairs in shuffled order (keeping the pairing)
			target_context_pairs = list(zip(target_graph_ids, target_context_subgraph_ids, subgraph_contexts_ids))
			shuffle(target_context_pairs)
			target_graph_ids, target_context_subgraph_ids, subgraph_contexts_ids= list(zip(*target_context_pairs))

			graph_targetSubgraph_subgraphContexts = [(graphTarget, subgraphTarget, subgraphContext) for (graphTarget, subgraphTarget, subgraphContext) in zip(target_graph_ids, target_context_subgraph_ids, subgraph_contexts_ids)]
			self.context_pair_dataset.append(graph_targetSubgraph_subgraphContexts)
		self.epoch_flag = False
		self.dataset_traverser = 0

	def __len__(self):
		# Return the number of total number of subgraphs
		# print("The number of graph-subgraph pairs considering batchsize in the dataset is")
		return self._subgraphcount

	def __getitem__(self, idx):
		if self.dataset_traverser == len(self.graph_fname_list):
			self.dataset_traverser = 0
			graph_targetSubgraph_subgraphContexts = self.context_pair_dataset[self.dataset_traverser]
			self.dataset_traverser += 1
			return graph_targetSubgraph_subgraphContexts
		else:
			graph_targetSubgraph_subgraphContexts = self.context_pair_dataset[self.dataset_traverser]
			self.dataset_traverser += 1
			return graph_targetSubgraph_subgraphContexts

	@staticmethod
	def collate(batches):
		all_targets = [target for batch in batches for target, _, _ in batch if len(batch)>0]
		all_contexts = [context for batch in batches for _, context, _ in batch if len(batch)>0]
		all_subgraph_contexts = [neg_context for batch in batches for _, _, neg_context in batch if len(batch)>0]

		return torch.LongTensor(all_targets), torch.LongTensor(all_contexts), torch.LongTensor(all_subgraph_contexts)


if __name__ == '__main__':
	corpus_dir = "../data/dortmund_gexf/MUTAG"

	# Normal Harddrive corpus
	corpus = Corpus(corpus_dir, extension=".wld2", max_files=60)
	# additional_file = "/home/morio/workspace/geo2dr/geometric2dr/file_handling/dortmund_gexf/MUTAG/99.gexf.wld2"
	# corpus.add_file(additional_file)
	dataloader = DataLoader(corpus, batch_size=4, shuffle=False, num_workers=0, collate_fn=corpus.collate)

	# In memory corpus
	in_mem_corpus = InMemoryCorpus(corpus_dir, extension=".wld2", max_files=60)
	mem_dataloader = DataLoader(in_mem_corpus, batch_size=4, shuffle=False, num_workers=0, collate_fn=corpus.collate)

	# In memory pvdm corpus
	in_mem_pvdm_corpus = PVDMCorpus(corpus_dir, extension=".awe_4_nodes", max_files=60, window_size=3)
	pvdm_dataloader = DataLoader(in_mem_pvdm_corpus, batch_size=4, shuffle=False, num_workers=0, collate_fn=corpus.collate)