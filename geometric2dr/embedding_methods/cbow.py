"""CBOW model with negative sampling as in Mikolov et al. [5]_. 

It is used with the corpus classes in cbow_data_reader which handles 
the data reading and loading. This allows construction of full CBOW
based systems. It is one of the choices of neural language model for
recreating DGK [2]_ like systems.

"""

# Author: Paul Scherer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Cbow(nn.Module):
	"""Pytorch implementation of the CBOW architecture with negative sampling
	as in Mikolov et al. [5]_

	This is used in DGK models for example to learn embeddings
	of substructures for downstream graph kernel definitions.

	Parameters
	----------
	num_targets : int
		The number of targets to embed. Typically the number of substructure
		patterns, but can be repurposed to be number of graphs. 
	vocab_size : int
		The size of the vocabulary; the number of unique substructure patterns
	embedding_dimension : int
		The desired dimensionality of the embeddings.

	Returns
	-------
	self : CBow 
		a torch.nn.Module of the CBOW model
	"""
	def __init__(self, num_targets, vocab_size, embedding_dimension):
		super(Cbow, self).__init__()
		self.num_targets = num_targets
		self.embedding_dimension = embedding_dimension
		self.vocab_size = vocab_size
		
		self.target_embeddings = nn.Embedding(num_targets, embedding_dimension) #D
		self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension) #W
		self.output_layer = nn.Embedding(vocab_size, embedding_dimension) #O

		# Xavier initialization of weights
		initrange = 1.0 / (self.embedding_dimension)
		init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
		init.constant_(self.context_embeddings.weight.data, 0)

		self.linear1 = nn.Linear(embedding_dimension, 128)
		self.activation1 = nn.ReLU()
		self.linear2 = nn.Linear(128, vocab_size)
		self.activation2 = nn.LogSoftmax(dim=-1)

	def forward(self, pos_target, pos_contexts, pos_negatives):
		"""Forward pass in network
		
		Parameters
		----------
		pos_target : torch.Long
			index of target embedding
		pos_contexts : torch.Long
			indices of context embeddings
		pos_negatives : torch.Long
			indices of negatives

		Returns
		-------
		torch.float
			the negative sampling loss
		
		"""
		# Be aware pos_contexts is typically several context embeddings
		# emb_target_graph = self.context_embeddings(pos_graph_emb)
		# mean_contexts_subgraphs = torch.mean(self.context_embeddings(pos_contexts), dim=1)

		# stack_target_contexts = torch.cat((emb_target_graph, mean_contexts_subgraphs), dim=1)

		# h = self.linear1(stack_target_contexts)
		# h = self.activation1(h)
		# h = self.linear2(h)
		# out = self.activation2(h)
		# return out

		# Negative sampling CBOW
		mean_contexts = torch.sum(self.context_embeddings(pos_contexts), dim=1)
		emb_context_target = self.output_layer(pos_target)
		emb_negative_targets = self.output_layer(pos_negatives)


		objective = torch.sum(torch.mul(mean_contexts, emb_context_target), dim=1) # mul is elementwise multiplication
		objective = torch.clamp(objective, max=10, min=-10)
		objective = -F.logsigmoid(objective)

		neg_objective = torch.bmm(emb_negative_targets, mean_contexts.unsqueeze(2)).squeeze()
		neg_objective = torch.clamp(neg_objective, max=10, min=-10)
		neg_objective = -torch.sum(F.logsigmoid(-neg_objective), dim=1)

		return torch.mean(objective+neg_objective)

	def give_target_embeddings(self):
		"""Return the target embeddings as a numpy matrix

		Returns
		-------
		numpy ndarray
		    Numpy vocab_size x emb_dimension matrix of substructure pattern embeddings
		"""

		embedding = self.target_embeddings.weight.cpu().data.numpy()
		return embedding