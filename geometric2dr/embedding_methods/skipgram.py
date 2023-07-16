"""General Skipgram model with negative sampling originally introduced by word2vec paper
Mikolov et al [5]_. Used by DGK [2]_ and Graph2Vec [4]_ to learn substructure and graph level embeddings 

It is used by the SkipgamCorpus and PVDBOWCorpus to build complete Skipgram and PVDBOW systems respectively.
SkipgramCorpus and PVDBOWCorpus are found in skipgram_data_reader and pvdbow_data_reader modules respectively

Author: Paul Scherer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Skipgram(nn.Module):
    """Pytorch implementation of the skipgram with negative
    sampling as in Mikolov et al. [5]_

    Based on the inputs it can be used as the skipgram described in the original Word2Vec paper [5]_ , or as
    Doc2Vec (PV-DBOW) in Le and Mikolov [6]_

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
    self : Skipgram
            a torch.nn.Module of the Skipgram model
    """

    def __init__(self, num_targets, vocab_size, embedding_dimension):
        super(Skipgram, self).__init__()
        self.num_targets = num_targets
        self.embedding_dimension = embedding_dimension
        self.target_embeddings = nn.Embedding(
            num_targets, embedding_dimension, sparse=True
        )  # its a tensor with a lookup function overloading the embedding(word) oart
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_dimension, sparse=True
        )

        # Xavier initialization of weights
        initrange = 1.0 / (self.embedding_dimension)
        init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.context_embeddings.weight.data, 0)

    def forward(self, pos_target, pos_context, neg_context):
        """Forward pass in network

        Parameters
        ----------
        pos_target : torch.Long
                index of target embedding
        pos_context : torch.Long
                index of context embedding
        neg_context : torch.Long
                index of negative

        Returns
        -------
        torch.float
                the negative sampling loss

        """
        emb_target = self.target_embeddings(pos_target)
        emb_context = self.context_embeddings(pos_context)
        emb_neg_context = self.context_embeddings(neg_context)

        objective = torch.sum(
            torch.mul(emb_target, emb_context), dim=1
        )  # mul is elementwise multiplication
        objective = torch.clamp(objective, max=10, min=-10)
        objective = -F.logsigmoid(objective)

        neg_objective = torch.bmm(emb_neg_context, emb_target.unsqueeze(2)).squeeze()
        neg_objective = torch.clamp(neg_objective, max=10, min=-10)
        neg_objective = -torch.sum(F.logsigmoid(-neg_objective), dim=1)

        return torch.mean(objective + neg_objective)

    def give_target_embeddings(self):
        embedding = self.target_embeddings.weight.cpu().data.numpy()
        return embedding
