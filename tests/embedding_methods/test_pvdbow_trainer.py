"""Unit tests for trainers

"""

import os
from pathlib import Path
from unittest import TestCase
from geometric2dr.embedding_methods.pvdbow_trainer import Trainer, InMemoryTrainer


class TestTrainer(TestCase):
    def setUp(self) -> None:
        corpus_data_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        self.output_embedding_fh = "PVDBOW_Embeddings.json"
        min_count_patterns = 0  # min number of occurrences to be considered in vocabulary of subgraph patterns

        self.trainer = Trainer(
            corpus_dir=corpus_data_dir,
            extension=extension,
            max_files=0,
            output_fh=self.output_embedding_fh,
            emb_dimension=32,
            batch_size=128,
            epochs=100,
            initial_lr=0.1,
            min_count=min_count_patterns,
        )

    def test_train(self) -> None:
        self.trainer.train()
        final_embeddings = self.trainer.skipgram.give_target_embeddings()
        assert final_embeddings.shape[0] >= 188
        assert final_embeddings.shape[1] == 32


class TestInMemoryTrainer(TestCase):
    def setUp(self) -> None:
        corpus_data_dir = "tests/test_data/MUTAG/"
        extension = ".wld2"
        self.output_embedding_fh = "PVDBOW_Embeddings.json"
        min_count_patterns = 0  # min number of occurrences to be considered in vocabulary of subgraph patterns

        self.trainer = InMemoryTrainer(
            corpus_dir=corpus_data_dir,
            extension=extension,
            max_files=0,
            output_fh=self.output_embedding_fh,
            emb_dimension=32,
            batch_size=128,
            epochs=100,
            initial_lr=0.1,
            min_count=min_count_patterns,
        )

    def test_train(self) -> None:
        self.trainer.train()
        final_embeddings = self.trainer.skipgram.give_target_embeddings()
        assert final_embeddings.shape[0] >= 188
        assert final_embeddings.shape[1] == 32
