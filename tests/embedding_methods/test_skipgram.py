"""
Unit tests for the skipgram class
"""
import torch
from unittest import TestCase
from geometric2dr.embedding_methods.skipgram import *

class TestSkipgram(TestCase):
    def setUp(self) -> None:
        self.num_targets = 188
        self.vocab_size = 10
        self.embedding_dimension = 64
        self.skipgram = Skipgram(self.num_targets, self.vocab_size, self.embedding_dimension)

    def test_forward(self) -> None:
        pos_target = torch.LongTensor([1,2,3])
        pos_context = torch.LongTensor([2,3,4])
        neg_context = torch.LongTensor([[2,3,3],[2,3,4],[2,4,2]])

        loss = self.skipgram(pos_target, pos_context, neg_context)
        assert loss > 0

    def test_give_target_embeddings(self) -> None:
        embedding = self.skipgram.give_target_embeddings()
        assert embedding.shape[0] == 188 and embedding.shape[1] == 64