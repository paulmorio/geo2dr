"""
PVDM model as used in AWE

Author: Paul Scherer 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PVDM(nn.Module):
    """
    PyTorch implmentation of the PVDM as in Le and Mikolov.
    """

    def __init__(self, num_targets, vocab_size, embedding_dimension):
        super(PVDM, self).__init__()
        self.num_targets = num_targets
        self.embedding_dimension = embedding_dimension
        # its a tensor with a lookup function overloading the embedding(word) oart
        self.target_embeddings = nn.Embedding(
            num_targets, embedding_dimension)
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_dimension)

        concat_dim = 2*embedding_dimension
        # Xavier initialization of weights
        initrange = 1.0 / (self.embedding_dimension)
        init.uniform_(self.target_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.context_embeddings.weight.data, 0)

        self.linear1 = nn.Linear(concat_dim, 128)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation2 = nn.LogSoftmax(dim=-1)

    def forward(self, pos_graph_emb, pos_context_target, pos_contexts):
        # Be aware pos_contexts is typically several context embeddings
        emb_target = self.target_embeddings(pos_graph_emb)
        mean_contexts = torch.sum(self.context_embeddings(pos_contexts), dim=1)
        emb_context_target = self.context_embeddings(pos_context_target)

        stack_target_contexts = torch.cat((emb_target, mean_contexts), dim=1)
        h = self.linear1(stack_target_contexts)
        h = self.activation1(h)
        h = self.linear2(h)
        out = self.activation2(h)
        return out


    def give_target_embeddings(self):
        embedding = self.target_embeddings.weight.cpu().data.numpy()
        return embedding