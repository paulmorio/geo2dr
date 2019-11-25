"""
Skipgram model

Uses negative sampling to learn better representations (and quicker too)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Skipgram(nn.Module):
	"""
	Pytorch implementation of the skipgram with negative 
	sampling as in Mikolov et al.
	"""

	def __init__(self, num_target_embs, vocab_size, embedding_dimension):
		super(skipgram, self).__init__()
		self.num_targets = num_targets
		self.embedding_dimension = embedding_dimension
		self.target_embeddings = nn.Embedding(num_targets, embedding_dimension, sparse=True) # its a tensor with a lookup function overloading the embedding(word) oart
		self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension, sparse=True)

		# Xavier initialization of weights
		initrange=1.0/(self.embedding_dimension+vocab_size) 
		init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
		init.constant_(self.context_embeddings.weight.data,0)

	def forward(self, pos_target, pos_context, neg_context):
		emb_target = self.target_embeddings(pos_target)
		emb_context = self.context_embeddings(pos_context)
		emb_neg_context = self.context_embeddings(neg_context)

		objective = torch.sum(torch.mul(emb_target, emb_context), dim=1) # mul is elementwise multiplication
		objective = torch.clamp(objective, max=10, min=-10)
		objective = -F.logsigmoid(objective)

		neg_objective = torch.bmm(emb_neg_context, emb_target.unsqueeze(2)).squeeze()
		neg_objective = torch.clamp(neg_objective, max=10, min=-10)
		neg_objective = -torch.sum(F.logsigmoid(-neg_objective), dim=1)

		return torch.mean(objective+neg_objective)

	def save_embedding(self, id2graph, file_name):
		embedding = self.target_embeddings.weight.cpu().data.numpy()
		with open(file_name, 'w') as f:
			f.write('%d %d\n' % (len(id2word), self.emb_dimension))
			for gid, g in id2graph.items():
				e = ' '.join(map(lambda x: str(x), embedding[gid]))
				f.write('%s %s\n' % (w,e))
